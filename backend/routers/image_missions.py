from __future__ import annotations

from exif_loader import SEKI_AGL_M
from exif_loader import load_image_dir
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from models import CameraConfig
from models import GPSCoordinate
from models import GimbalMode
from pathlib import Path
from pydantic import BaseModel
import io
import json
import math
import shutil
import tempfile
import zipfile
from backend.config import BASE_DIR, MAX_IMAGE_UPLOAD_BYTES, MAX_IMAGE_UPLOAD_FILES, TASK3_UPLOADS_DIR
from backend.services.images import _build_compatible_image_stitcher, _crop_nonzero_border, _detect_clusters, _image_mosaic_projection_data, _image_row_to_footprint, _normalise_mission_id, _prepare_image_preview_inputs
from backend.services.telemetry import _build_projection_mosaic, _sanitize_for_json
from backend.state import _image_missions, _image_preview_cache

router = APIRouter()


class ImageLoadRequest(BaseModel):
    path: str = "task3_data/seki_southfork/images"
    mission_id: str = "SEKI_SOUTHFORK"
    agl_m: float = SEKI_AGL_M


@router.post("/image_missions/load")
async def load_image_mission(req: ImageLoadRequest):
    """
    Load a directory of geotagged JPGs into the image-mission store.

    Extracts EXIF GPS from each JPG, derives headings from consecutive
    positions, and stores the result under req.mission_id.
    Existing CSV/SRT missions in _missions are not affected.
    """
    dir_path = Path(req.path)
    if not dir_path.is_absolute():
        dir_path = BASE_DIR / req.path

    try:
        dir_path = dir_path.resolve()
        if not dir_path.is_relative_to(BASE_DIR.resolve()):
            raise ValueError()
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=403, detail="Access to this directory is forbidden.")

    if not dir_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Image directory not found: {dir_path}",
        )

    rows = load_image_dir(dir_path, agl_m=req.agl_m)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No valid geotagged images found in directory.",
        )

    _image_missions[req.mission_id] = rows
    stale_keys = [k for k in _image_preview_cache if len(k) > 1 and k[1] == req.mission_id]
    for key in stale_keys:
        _image_preview_cache.pop(key, None)
    return {"status": "ok", "mission_id": req.mission_id, "image_count": len(rows)}


@router.post("/image_missions/upload")
async def upload_image_mission(
    files: list[UploadFile] = File(...),
    mission_id: str = Form("UPLOAD"),
    agl_m: float = Form(SEKI_AGL_M),
):
    """
    Upload geotagged JPG files from the local computer and register them as a
    new image mission.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No image files were provided.")
    if len(files) > MAX_IMAGE_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"Too many files (max {MAX_IMAGE_UPLOAD_FILES}).")
    if not math.isfinite(agl_m) or agl_m <= 0:
        raise HTTPException(status_code=400, detail="agl_m must be finite and positive.")

    normalised_mission_id = _normalise_mission_id(mission_id, fallback_prefix="UPLOAD")
    TASK3_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    mission_dir = TASK3_UPLOADS_DIR / normalised_mission_id

    # Validate the complete replacement before touching the existing mission.
    with tempfile.TemporaryDirectory(prefix=".incoming_", dir=TASK3_UPLOADS_DIR) as staging:
        staging_dir = Path(staging)
        saved_count = 0
        saved_bytes = 0
        for index, upload in enumerate(files):
            filename = Path(upload.filename or f"image_{index + 1}.jpg").name
            if Path(filename).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            target_path = staging_dir / filename
            if target_path.exists():
                raise HTTPException(status_code=400, detail="Image filenames must be unique.")
            with target_path.open("wb") as out_fh:
                shutil.copyfileobj(upload.file, out_fh)
            saved_bytes += target_path.stat().st_size
            if saved_bytes > MAX_IMAGE_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Upload too large (max 2 GB).")
            saved_count += 1

        if saved_count == 0:
            raise HTTPException(status_code=400, detail="Upload must include JPG images.")
        rows = load_image_dir(staging_dir, agl_m=agl_m)
        if not rows:
            raise HTTPException(status_code=400, detail="No valid geotagged JPG images were found in the uploaded files.")

        # Keep a rollback copy on the same filesystem until the rename succeeds.
        with tempfile.TemporaryDirectory(prefix=".previous_", dir=TASK3_UPLOADS_DIR) as previous:
            backup_dir = Path(previous) / "mission"
            had_previous = mission_dir.exists()
            if had_previous:
                mission_dir.rename(backup_dir)
            try:
                staging_dir.rename(mission_dir)
            except OSError:
                if had_previous:
                    backup_dir.rename(mission_dir)
                raise
        rows = [{**row, "path": str(mission_dir / Path(row["path"]).name)} for row in rows]

    _image_missions[normalised_mission_id] = rows
    stale_keys = [key for key in _image_preview_cache if len(key) > 1 and key[1] == normalised_mission_id]
    for key in stale_keys:
        _image_preview_cache.pop(key, None)

    return {
        "status": "ok",
        "mission_id": normalised_mission_id,
        "image_count": len(rows),
        "uploaded_files": saved_count,
        "path": str(mission_dir),
    }


@router.get("/image_missions")
async def list_image_missions():
    """List all loaded image missions (separate from CSV/SRT /missions)."""
    return [
        {"mission_id": mid, "image_count": len(rows), "source": "exif"}
        for mid, rows in _image_missions.items()
        if rows
    ]


@router.get("/image_mission/{mission_id}/footprints")
async def get_image_footprints(mission_id: str):
    """
    Task 1 applied to image missions: return footprint polygons for every image.

    Heading is derived from consecutive GPS positions (not hardcoded to 0 deg).
    """
    rows = _image_missions.get(mission_id)
    if rows is None:
        raise HTTPException(
            status_code=404, detail=f"Image mission '{mission_id}' not found."
        )

    footprints = [fp for row in rows if (fp := _image_row_to_footprint(row)) is not None]
    return {"mission_id": mission_id, "count": len(footprints), "footprints": footprints}


@router.get("/image_mission/{mission_id}/mosaic")
async def get_image_mosaic_metadata(mission_id: str, image_width_px: int = 4928):
    """
    Task 3: return mosaic canvas metadata for an image mission.

    Uses actual sensor resolution (default 4928 px) and real per-image headings.
    Response shape matches /mission/{id}/mosaic for consistency, with
    status='exif_metadata' to distinguish from telemetry-only missions.
    """
    rows = _image_missions.get(mission_id)
    if rows is None:
        raise HTTPException(
            status_code=404, detail=f"Image mission '{mission_id}' not found."
        )

    from footprint_calculator import FootprintResult

    footprints: list[FootprintResult] = []
    for row in rows:
        fp_dict = _image_row_to_footprint(row)
        if fp_dict is None:
            continue
        corners = [GPSCoordinate(lat=c["lat"], lon=c["lon"]) for c in fp_dict["corners"]]
        footprints.append(FootprintResult(
            top_left=corners[0], top_right=corners[1],
            bottom_right=corners[2], bottom_left=corners[3],
            nadir=GPSCoordinate(lat=fp_dict["nadir"]["lat"], lon=fp_dict["nadir"]["lon"]),
            ground_width_m=fp_dict["width_m"], ground_height_m=fp_dict["height_m"],
        ))

    if not footprints:
        raise HTTPException(
            status_code=422,
            detail="No valid footprints — cannot compute mosaic bounds.",
        )

    all_lats = [c.lat for fp in footprints for c in fp.corners]
    all_lons = [c.lon for fp in footprints for c in fp.corners]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)

    METRES_PER_LAT_DEG = 111_320.0
    avg_lat_rad = math.radians((lat_min + lat_max) / 2.0)
    gsd_m = min(fp.ground_width_m / max(image_width_px, 1) for fp in footprints)

    lat_span_m = (lat_max - lat_min) * METRES_PER_LAT_DEG
    lon_span_m = (lon_max - lon_min) * METRES_PER_LAT_DEG * math.cos(avg_lat_rad)
    W_canvas = max(1, round(lon_span_m / gsd_m))
    H_canvas = max(1, round(lat_span_m / gsd_m))

    return {
        "mission_id":      mission_id,
        "status":          "exif_metadata",
        "message":         (
            "Mosaic metadata computed from EXIF GPS footprints. "
            "Call /image_mission/{id}/mosaic/preview for a downsampled stitched image."
        ),
        "footprint_count": len(footprints),
        "image_count":     len(rows),
        "bounding_box": {
            "TL": {"lat": lat_max, "lon": lon_min},
            "TR": {"lat": lat_max, "lon": lon_max},
            "BR": {"lat": lat_min, "lon": lon_max},
            "BL": {"lat": lat_min, "lon": lon_min},
        },
        "gsd_m":           round(gsd_m, 6),
        "canvas_width_px": W_canvas,
        "canvas_height_px": H_canvas,
        "image_width_px":  image_width_px,
    }


@router.get("/image_mission/{mission_id}/project")
async def project_image_pixel(
    mission_id: str,
    x: int,
    y: int,
    image_width_px: int = 4928,
):
    """
    Task 4: Convert a mosaic pixel coordinate to a GPS position (image mission).

    x, y are 0-indexed pixel offsets into the mosaic canvas (0 = top-left).
    Use the same image_width_px as /image_mission/{id}/mosaic to stay consistent;
    the canvas_width_px and canvas_height_px returned here match that route.

    Returns 404 if the mission is missing, 422 if it has no valid footprints or
    the canvas is degenerate, and 400 if x or y falls outside the canvas bounds.
    """
    lat_min, lat_max, lon_min, lon_max, gsd_m, W_canvas, H_canvas = (
        _image_mosaic_projection_data(mission_id, image_width_px)
    )

    from drone_oversight import GPSProjector

    mosaic = _build_projection_mosaic(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        gsd_m=gsd_m,
        width_px=W_canvas,
        height_px=H_canvas,
    )

    try:
        coord = GPSProjector(mosaic).pixel_to_gps(x, y)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "mission_id":      mission_id,
        "x":               x,
        "y":               y,
        "lat":             coord.lat,
        "lon":             coord.lon,
        "canvas_width_px": W_canvas,
        "canvas_height_px": H_canvas,
        "gsd_m":           round(gsd_m, 6),
    }


@router.get("/image_mission/{mission_id}/project/reverse")
async def project_image_gps_to_pixel(
    mission_id: str,
    lat: float,
    lon: float,
    image_width_px: int = 4928,
):
    """
    Task 4: Convert a GPS coordinate inside an image mission mosaic to a pixel.
    """
    lat_min, lat_max, lon_min, lon_max, gsd_m, W_canvas, H_canvas = (
        _image_mosaic_projection_data(mission_id, image_width_px)
    )

    from drone_oversight import GPSProjector

    mosaic = _build_projection_mosaic(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        gsd_m=gsd_m,
        width_px=W_canvas,
        height_px=H_canvas,
    )

    try:
        x, y = GPSProjector(mosaic).gps_to_pixel(GPSCoordinate(lat=lat, lon=lon))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "mission_id": mission_id,
        "lat": lat,
        "lon": lon,
        "x": x,
        "y": y,
        "canvas_width_px": W_canvas,
        "canvas_height_px": H_canvas,
        "gsd_m": round(gsd_m, 6),
    }


@router.get("/image_mission/{mission_id}/mosaic/preview")
async def get_image_mosaic_preview(
    mission_id: str,
    scale: float = 0.05,
    refine: bool = False,
    normalize_altitude: bool = True,
    force: bool = False,
    use_window_merge: bool = False,
    use_alignment_refinement: bool = False,
    window_size: int = 8,
    window_stride: int = 4,
    use_sba: bool = False,
    use_semantic_seamline: bool = False,
    use_novelty_composite: bool = True,
    novelty_max_overlap_ratio: float = 0.90,
    novelty_min_pixels: int = 1500,
    novelty_gps_min_move_m: float = 2.0,
    max_keyframes: int = 0,
    keyframe_latest_wins: bool = True,
    novelty_angle_threshold_deg: float = 12.0,
    novelty_min_overlap_with_last: float = 0.30,
):
    """
    Task 3: Generate a downscaled PNG preview of the stitched mosaic.

    Each image is resized by `scale` before stitching so the canvas stays
    in memory (default scale=0.05 → ~18 MB canvas, ~1-3 MB PNG output).
    Raises 404 if mission not loaded, 422 if <2 valid images can be opened.

    Parameters
    ----------
    scale : float   Resize factor per axis (clamped to [0.01, 0.20]).
                    0.05 → 246×163 px per SEKI image, ~2950×2050 px canvas.
    refine : bool   Experimental feature-refinement path. Default False; the
                    supported preview mode remains GPS-guided placement.
    use_sba : bool  Enable Structural Bundle Adjustment (LoFTR + line constraints,
                    requires torch/kornia).  Default False.
    use_semantic_seamline : bool  Enable Semantic Seamline Detection and
                    Laplacian pyramid blending.  Default False.
    """
    rows = _image_missions.get(mission_id)
    if rows is None:
        raise HTTPException(
            status_code=404, detail=f"Image mission '{mission_id}' not found."
        )

    scale = max(0.01, min(0.20, scale))
    if refine and not use_alignment_refinement:
        use_alignment_refinement = True
    if refine and scale < 0.12:
        scale = 0.12
    # Extremely small preview scales blur imagery and disable refine/SBA quality.
    if refine and scale < 0.10:
        scale = 0.10
    if use_sba and scale < 0.10:
        scale = 0.10
    cache_key = (
        "single",
        mission_id,
        round(scale, 4),
        bool(refine),
        bool(normalize_altitude),
        bool(use_window_merge),
        bool(use_alignment_refinement),
        max(1, int(window_size)),
        max(1, int(window_stride)),
        bool(use_sba),
        bool(use_semantic_seamline),
        bool(use_novelty_composite),
        round(float(novelty_max_overlap_ratio), 4),
        int(max(1, novelty_min_pixels)),
        round(float(novelty_gps_min_move_m), 3),
        int(max_keyframes),
        bool(keyframe_latest_wins),
        round(float(novelty_angle_threshold_deg), 3),
        round(float(novelty_min_overlap_with_last), 4),
    )
    if not force:
        cached = _image_preview_cache.get(cache_key)
        if cached is not None:
            return Response(content=cached, media_type="image/png")

    try:
        from PIL import Image as PILImage
        from drone_oversight import ImageStitcher
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")

    images, footprints_for_stitch = _prepare_image_preview_inputs(
        rows, scale=scale, normalize_altitude=normalize_altitude
    )

    if len(images) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 loadable images to stitch; got {len(images)}.",
        )

    camera_config = CameraConfig(
        fov_h=rows[0]["fov_h"],
        fov_v=rows[0]["fov_v"],
        gimbal_mode=GimbalMode.NADIR,
    )

    mosaic = _build_compatible_image_stitcher(
        ImageStitcher,
        images,
        footprints_for_stitch,
        camera_config,
        refine=refine,
        composite_mode="sharpest",
        window_size=max(1, int(window_size)),
        window_stride=max(1, int(window_stride)),
        compute_window_mosaics=bool(use_window_merge),
        use_window_merge=bool(use_window_merge),
        use_alignment_refinement=bool(use_alignment_refinement),
        use_sba=bool(use_sba),
        use_semantic_seamline=bool(use_semantic_seamline),
        sba_weights={"w1": 4.0, "w2": 1.5, "w3": 0.0, "w4": 0.03} if use_sba else None,
        use_novelty_composite=bool(use_novelty_composite),
        novelty_max_overlap_ratio=float(max(0.0, min(0.99, novelty_max_overlap_ratio))),
        novelty_min_pixels=int(max(1, novelty_min_pixels)),
        novelty_gps_min_move_m=float(max(0.0, novelty_gps_min_move_m)),
        max_keyframes=(None if int(max_keyframes) <= 0 else int(max_keyframes)),
        keyframe_latest_wins=bool(keyframe_latest_wins),
        novelty_angle_threshold_deg=float(max(0.0, novelty_angle_threshold_deg)),
        novelty_min_overlap_with_last=float(max(0.0, min(1.0, novelty_min_overlap_with_last))),
    ).stitch()

    buf = io.BytesIO()
    out_img = _crop_nonzero_border(mosaic.image)
    PILImage.fromarray(out_img).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    _image_preview_cache[cache_key] = png_bytes
    return Response(content=png_bytes, media_type="image/png")


@router.get("/image_mission/{mission_id}/mosaic/export")
async def export_image_mosaic(
    mission_id: str,
    scale: float = 0.05,
    refine: bool = False,
    normalize_altitude: bool = True,
    use_window_merge: bool = False,
    use_alignment_refinement: bool = False,
    window_size: int = 8,
    window_stride: int = 4,
    use_sba: bool = False,
    use_semantic_seamline: bool = False,
    use_novelty_composite: bool = True,
    novelty_max_overlap_ratio: float = 0.90,
    novelty_min_pixels: int = 1500,
    novelty_gps_min_move_m: float = 2.0,
    max_keyframes: int = 0,
    keyframe_latest_wins: bool = True,
    novelty_angle_threshold_deg: float = 12.0,
    novelty_min_overlap_with_last: float = 0.30,
):
    """
    Task 3: export the current stitched mosaic as a delivery package.

    This route reuses the existing image-mission stitching pipeline and returns
    a ZIP containing:
      - mosaic.png
      - mosaic.json  (geographic metadata)
      - mosaic.pgw   (world file in geographic degrees)

    The stitching algorithm itself is unchanged; this is packaging around the
    existing stitch output.
    """
    rows = _image_missions.get(mission_id)
    if rows is None:
        raise HTTPException(
            status_code=404, detail=f"Image mission '{mission_id}' not found."
        )

    scale = max(0.01, min(0.20, scale))
    if refine and not use_alignment_refinement:
        use_alignment_refinement = True
    if refine and scale < 0.12:
        scale = 0.12
    if use_sba and scale < 0.10:
        scale = 0.10

    try:
        from PIL import Image as PILImage
        from drone_oversight import ImageStitcher
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")

    images, footprints_for_stitch = _prepare_image_preview_inputs(
        rows, scale=scale, normalize_altitude=normalize_altitude
    )

    if len(images) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 loadable images to stitch; got {len(images)}.",
        )

    camera_config = CameraConfig(
        fov_h=rows[0]["fov_h"],
        fov_v=rows[0]["fov_v"],
        gimbal_mode=GimbalMode.NADIR,
    )

    mosaic = _build_compatible_image_stitcher(
        ImageStitcher,
        images,
        footprints_for_stitch,
        camera_config,
        refine=refine,
        composite_mode="sharpest",
        window_size=max(1, int(window_size)),
        window_stride=max(1, int(window_stride)),
        compute_window_mosaics=bool(use_window_merge),
        use_window_merge=bool(use_window_merge),
        use_alignment_refinement=bool(use_alignment_refinement),
        use_sba=bool(use_sba),
        use_semantic_seamline=bool(use_semantic_seamline),
        sba_weights={"w1": 4.0, "w2": 1.5, "w3": 0.0, "w4": 0.03} if use_sba else None,
        use_novelty_composite=bool(use_novelty_composite),
        novelty_max_overlap_ratio=float(max(0.0, min(0.99, novelty_max_overlap_ratio))),
        novelty_min_pixels=int(max(1, novelty_min_pixels)),
        novelty_gps_min_move_m=float(max(0.0, novelty_gps_min_move_m)),
        max_keyframes=(None if int(max_keyframes) <= 0 else int(max_keyframes)),
        keyframe_latest_wins=bool(keyframe_latest_wins),
        novelty_angle_threshold_deg=float(max(0.0, novelty_angle_threshold_deg)),
        novelty_min_overlap_with_last=float(max(0.0, min(1.0, novelty_min_overlap_with_last))),
    ).stitch()

    image = mosaic.image
    height_px, width_px = image.shape[:2]
    if width_px < 2 or height_px < 2:
        raise HTTPException(
            status_code=422,
            detail="Stitched mosaic canvas is degenerate; cannot export georeferenced package.",
        )

    tl, tr, br, bl = mosaic.bounding_box
    lon_per_px = (tr.lon - tl.lon) / max(width_px - 1, 1)
    lat_per_px = (bl.lat - tl.lat) / max(height_px - 1, 1)
    worldfile = "\n".join([
        f"{lon_per_px:.12f}",
        "0.0",
        "0.0",
        f"{lat_per_px:.12f}",
        f"{tl.lon:.12f}",
        f"{tl.lat:.12f}",
    ]) + "\n"

    metadata = {
        "mission_id": mission_id,
        "format": "png+worldfile+json",
        "scale": scale,
        "refine": bool(refine),
        "normalize_altitude": bool(normalize_altitude),
        "image_width_px": width_px,
        "image_height_px": height_px,
        "gsd_m": round(mosaic.gsd_m, 6),
        "bounding_box": {
            "TL": {"lat": tl.lat, "lon": tl.lon},
            "TR": {"lat": tr.lat, "lon": tr.lon},
            "BR": {"lat": br.lat, "lon": br.lon},
            "BL": {"lat": bl.lat, "lon": bl.lon},
        },
    }

    png_buf = io.BytesIO()
    PILImage.fromarray(image).save(png_buf, format="PNG")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mosaic.png", png_buf.getvalue())
        zf.writestr("mosaic.json", json.dumps(metadata, indent=2))
        zf.writestr("mosaic.pgw", worldfile)

    filename = f"{mission_id.lower()}_mosaic_export.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers=headers,
    )


@router.get("/image_mission/{mission_id}/mosaic/debug")
async def get_image_mosaic_debug(
    mission_id: str,
    scale: float = 0.05,
    refine: bool = True,
    normalize_altitude: bool = True,
    use_window_merge: bool = False,
    use_alignment_refinement: bool = False,
    window_size: int = 8,
    window_stride: int = 4,
    use_sba: bool = False,
    use_semantic_seamline: bool = False,
    use_novelty_composite: bool = True,
    novelty_max_overlap_ratio: float = 0.90,
    novelty_min_pixels: int = 1500,
    novelty_gps_min_move_m: float = 2.0,
    max_keyframes: int = 0,
    keyframe_latest_wins: bool = True,
    novelty_angle_threshold_deg: float = 12.0,
    novelty_min_overlap_with_last: float = 0.30,
):
    """
    Task 3 diagnostics: return stitch/refine metrics without serving an image.

    This endpoint exists to debug real-world mosaic quality issues on image
    missions by exposing refine pair metrics, acceptance/fallback counts, and
    basic canvas coverage numbers for the current preview settings.
    """
    rows = _image_missions.get(mission_id)
    if rows is None:
        raise HTTPException(
            status_code=404, detail=f"Image mission '{mission_id}' not found."
        )

    scale = max(0.01, min(0.20, scale))
    if not use_novelty_composite:
        if not refine:
            refine = True
        if not use_alignment_refinement:
            use_alignment_refinement = True
        if scale < 0.12:
            scale = 0.12
    if refine and scale < 0.10:
        scale = 0.10
    if use_sba and scale < 0.10:
        scale = 0.10

    try:
        from drone_oversight import ImageStitcher
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")

    images, footprints_for_stitch = _prepare_image_preview_inputs(
        rows, scale=scale, normalize_altitude=normalize_altitude
    )
    if len(images) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 loadable images to stitch; got {len(images)}.",
        )

    camera_config = CameraConfig(
        fov_h=rows[0]["fov_h"],
        fov_v=rows[0]["fov_v"],
        gimbal_mode=GimbalMode.NADIR,
    )
    stitcher = _build_compatible_image_stitcher(
        ImageStitcher,
        images,
        footprints_for_stitch,
        camera_config,
        refine=refine,
        composite_mode="sharpest",
        window_size=max(1, int(window_size)),
        window_stride=max(1, int(window_stride)),
        compute_window_mosaics=True,
        use_window_merge=bool(use_window_merge),
        use_alignment_refinement=bool(use_alignment_refinement),
        use_sba=bool(use_sba),
        use_semantic_seamline=bool(use_semantic_seamline),
        sba_weights={"w1": 4.0, "w2": 1.5, "w3": 0.0, "w4": 0.03} if use_sba else None,
        use_novelty_composite=bool(use_novelty_composite),
        novelty_max_overlap_ratio=float(max(0.0, min(0.99, novelty_max_overlap_ratio))),
        novelty_min_pixels=int(max(1, novelty_min_pixels)),
        novelty_gps_min_move_m=float(max(0.0, novelty_gps_min_move_m)),
        max_keyframes=(None if int(max_keyframes) <= 0 else int(max_keyframes)),
        keyframe_latest_wins=bool(keyframe_latest_wins),
        novelty_angle_threshold_deg=float(max(0.0, novelty_angle_threshold_deg)),
        novelty_min_overlap_with_last=float(max(0.0, min(1.0, novelty_min_overlap_with_last))),
    )
    mosaic = stitcher.stitch()

    return JSONResponse({
        "mission_id": mission_id,
        "scale": scale,
        "normalize_altitude": bool(normalize_altitude),
        "loaded_image_count": len(images),
        "source_row_count": len(rows),
        "gsd_m": round(mosaic.gsd_m, 6),
        "diagnostics": _sanitize_for_json(mosaic.diagnostics),
    })


@router.get("/image_mission/{mission_id}/mosaic/preview/grouped")
async def get_image_mosaic_preview_grouped(
    mission_id: str,
    scale: float = 0.05,
    gap_m: float = 200.0,
    refine: bool = False,
    normalize_altitude: bool = True,
    force: bool = False,
):
    """
    Task 3: Generate a side-by-side PNG preview with one panel per spatial cluster.

    Splits images into contiguous groups using *gap_m* as the separation
    threshold, stitches each group independently, and returns a single PNG
    with cluster panels arranged left-to-right, separated by a 10-pixel
    gray border.

    The existing /mosaic/preview endpoint is not affected.

    Parameters
    ----------
    scale : float   Per-axis resize factor before stitching (clamped [0.01, 0.20]).
    gap_m : float   Gap threshold for cluster detection in metres (default 200 m).
    refine : bool   Experimental feature-refinement path for each cluster.
                    Default False; grouped GPS-guided preview is the supported mode.
    """
    rows = _image_missions.get(mission_id)
    if rows is None:
        raise HTTPException(
            status_code=404, detail=f"Image mission '{mission_id}' not found."
        )

    scale = max(0.01, min(0.20, scale))
    gap_m = max(1.0, gap_m)
    cache_key = ("grouped", mission_id, round(scale, 4), round(gap_m, 1), bool(refine), bool(normalize_altitude))
    if not force:
        cached = _image_preview_cache.get(cache_key)
        if cached is not None:
            return Response(content=cached, media_type="image/png")

    try:
        from PIL import Image as PILImage
        from drone_oversight import ImageStitcher
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")

    clusters = _detect_clusters(rows, gap_threshold_m=gap_m)

    cluster_pil_images: list = []

    for cluster_rows in clusters:
        images, footprints_for_stitch = _prepare_image_preview_inputs(
            cluster_rows, scale=scale, normalize_altitude=normalize_altitude
        )

        if len(images) < 2:
            continue  # skip clusters too small to stitch

        camera_config = CameraConfig(
            fov_h=cluster_rows[0]["fov_h"],
            fov_v=cluster_rows[0]["fov_v"],
            gimbal_mode=GimbalMode.NADIR,
        )
        mosaic = _build_compatible_image_stitcher(
            ImageStitcher,
            images,
            footprints_for_stitch,
            camera_config,
            refine=refine,
            composite_mode="sharpest",
        ).stitch()
        cluster_pil_images.append(PILImage.fromarray(mosaic.image))

    if not cluster_pil_images:
        raise HTTPException(
            status_code=422,
            detail="No clusters produced a stitchable image (need ≥2 loadable images per cluster).",
        )

    if len(cluster_pil_images) == 1:
        final_img = cluster_pil_images[0]
    else:
        GAP_PX = 10
        total_w = sum(img.width for img in cluster_pil_images) + GAP_PX * (len(cluster_pil_images) - 1)
        max_h = max(img.height for img in cluster_pil_images)
        final_img = PILImage.new("RGB", (total_w, max_h), (180, 180, 180))
        x = 0
        for img in cluster_pil_images:
            final_img.paste(img, (x, 0))
            x += img.width + GAP_PX

    buf = io.BytesIO()
    final_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    _image_preview_cache[cache_key] = png_bytes
    return Response(content=png_bytes, media_type="image/png")
