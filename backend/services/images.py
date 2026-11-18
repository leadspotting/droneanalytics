from __future__ import annotations

import progress

from exif_loader import load_image_dir
from fastapi import HTTPException
from footprint_calculator import FootprintCalculator
from models import CameraConfig
from models import DroneTelemetry
from models import GPSCoordinate
from models import GimbalMode
from pathlib import Path
from typing import Optional
import math
import re
import time
from backend.config import BASE_DIR, _DEFAULT_IMAGE_MISSION_DIRS
from backend.state import _image_missions


def _crop_nonzero_border(image):
    """Crop fully-black borders from an RGB mosaic image."""
    import numpy as np
    if image is None or image.size == 0:
        return image
    mask = np.any(image != 0, axis=2)
    if not np.any(mask):
        return image
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return image[y0:y1, x0:x1]


def _autoload_image_missions() -> None:
    if _image_missions:
        return

    for mission_id, (rel_path, agl_m) in _DEFAULT_IMAGE_MISSION_DIRS.items():
        dir_path = BASE_DIR / rel_path
        if not dir_path.exists():
            continue
        try:
            rows = load_image_dir(dir_path, agl_m=agl_m)
        except Exception:
            continue
        if rows:
            _image_missions[mission_id] = rows


def _normalise_mission_id(raw_value: str | None, fallback_prefix: str = "UPLOAD") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (raw_value or "").strip()).strip("_").upper()
    if cleaned:
        return cleaned
    return f"{fallback_prefix}_{int(time.time())}"


def _image_row_to_footprint(row: dict) -> Optional[dict]:
    """
    Compute a footprint dict from one EXIF image row.

    Uses the row's actual heading (derived from consecutive GPS positions)
    and the per-row camera FOV — local to this image-mission flow.
    """
    try:
        from footprint_calculator import FootprintResult  # noqa: F401 (side-effect import ok)

        telemetry = DroneTelemetry(
            position=GPSCoordinate(lat=row["lat"], lon=row["lon"], alt=row["agl_m"]),
            heading=row["heading"],
            camera=CameraConfig(
                fov_h=row["fov_h"],
                fov_v=row["fov_v"],
                gimbal_mode=GimbalMode.NADIR,
            ),
        )
        result = FootprintCalculator(telemetry).compute()
        return {
            "nadir":    {"lat": row["lat"], "lon": row["lon"]},
            "corners":  [{"lat": c.lat, "lon": c.lon} for c in result.corners],
            "width_m":  round(result.ground_width_m, 1),
            "height_m": round(result.ground_height_m, 1),
            "altitude_m": row["agl_m"],
            "timestamp":  row.get("timestamp", ""),
            "filename":   Path(row["path"]).name,
        }
    except Exception:
        return None


def _prepare_image_preview_inputs(
    rows: list[dict],
    scale: float,
    normalize_altitude: bool = True,
):
    """
    Load preview-sized image arrays and matching footprints for an image mission.

    If normalize_altitude is enabled, apply Task 5 scaling before stitching so
    images captured higher/lower than the reference altitude are resized toward
    a common reference GSD.
    """
    from PIL import Image as PILImage
    import numpy as np
    from drone_oversight import AltitudeScaler
    from footprint_calculator import FootprintResult

    images: list[np.ndarray] = []
    footprints_for_stitch: list[FootprintResult] = []

    positive_alts = [float(r["agl_m"]) for r in rows if float(r.get("agl_m", 0.0)) > 0]
    reference_altitude_m = min(positive_alts) if positive_alts else 1.0
    reference_gsd_m = (
        2.0
        * reference_altitude_m
        * math.tan(math.radians(rows[0]["fov_h"] / 2.0))
        / max(rows[0]["width_px"], 1)
    )
    scaler = AltitudeScaler(
        reference_altitude_m=reference_altitude_m,
        reference_gsd_m=reference_gsd_m,
        camera_config=CameraConfig(
            fov_h=rows[0]["fov_h"],
            fov_v=rows[0]["fov_v"],
            gimbal_mode=GimbalMode.NADIR,
        ),
    )

    for _idx, row in enumerate(rows):
        progress.report(0.4 * _idx / max(len(rows), 1), f"Loading images {_idx + 1}/{len(rows)}")
        fp_dict = _image_row_to_footprint(row)
        if fp_dict is None:
            continue
        try:
            img_pil = PILImage.open(row["path"]).convert("RGB")
            altitude_scale = (
                scaler.compute_scale_factor(float(row["agl_m"]))
                if normalize_altitude
                else 1.0
            )
            effective_scale = max(0.005, min(0.5, scale * altitude_scale))
            new_w = max(1, round(img_pil.width * effective_scale))
            new_h = max(1, round(img_pil.height * effective_scale))
            img_pil = img_pil.resize((new_w, new_h), PILImage.BILINEAR)
            images.append(np.array(img_pil, dtype=np.uint8))
        except Exception:
            continue

        corners = [GPSCoordinate(lat=c["lat"], lon=c["lon"]) for c in fp_dict["corners"]]
        footprints_for_stitch.append(FootprintResult(
            top_left=corners[0], top_right=corners[1],
            bottom_right=corners[2], bottom_left=corners[3],
            nadir=GPSCoordinate(lat=fp_dict["nadir"]["lat"], lon=fp_dict["nadir"]["lon"]),
            ground_width_m=fp_dict["width_m"],
            ground_height_m=fp_dict["height_m"],
        ))

    progress.report(0.4, "Images loaded")
    return images, footprints_for_stitch


def _build_compatible_image_stitcher(
    image_stitcher_cls,
    images: list,
    footprints_for_stitch: list,
    camera_config: CameraConfig,
    refine: bool = False,
    **kwargs,
):
    """
    Build the Task 3 stitcher, passing all advanced kwargs through.

    ImageStitcher now accepts the full set of advanced pipeline flags via
    **_ignored_kwargs so unknown flags never crash the constructor.  All
    flags exposed by the preview / export / debug routes are forwarded here.
    """
    return image_stitcher_cls(
        images,
        footprints_for_stitch,
        camera_config,
        refine=refine,
        **kwargs,
    )


def _image_mosaic_projection_data(mission_id: str, image_width_px: int) -> tuple:
    """
    Reconstruct mosaic bounding box and canvas dimensions from image mission footprints.

    Raises HTTPException 404 / 422 as appropriate.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max, gsd_m, W_canvas, H_canvas)
    """
    from footprint_calculator import FootprintResult

    rows = _image_missions.get(mission_id)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"Image mission '{mission_id}' not found.")

    if image_width_px <= 0:
        raise HTTPException(
            status_code=400,
            detail="image_width_px must be a positive integer.",
        )

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
    gsd_m = min(fp.ground_width_m / image_width_px for fp in footprints)

    lat_span_m = (lat_max - lat_min) * METRES_PER_LAT_DEG
    lon_span_m = (lon_max - lon_min) * METRES_PER_LAT_DEG * math.cos(avg_lat_rad)
    W_canvas = max(1, round(lon_span_m / gsd_m))
    H_canvas = max(1, round(lat_span_m / gsd_m))

    if W_canvas < 2 or H_canvas < 2:
        raise HTTPException(
            status_code=422,
            detail="Canvas is degenerate (width or height < 2 px) — "
                   "footprints too sparse or image_width_px too small.",
        )

    return lat_min, lat_max, lon_min, lon_max, gsd_m, W_canvas, H_canvas


def _detect_clusters(
    rows: list[dict], gap_threshold_m: float = 200.0
) -> list[list[dict]]:
    """
    Split image-mission rows into spatially contiguous groups.

    A new cluster starts whenever the straight-line distance between
    consecutive frame positions exceeds *gap_threshold_m*.

    Parameters
    ----------
    rows            : list of image-row dicts with 'lat' and 'lon' keys.
    gap_threshold_m : distance threshold in metres (default 200 m).
                      SEKI intra-group steps are 4–8 m; the cross-group
                      gap is ~837 m, so 200 m sits safely between them.

    Returns
    -------
    list[list[dict]]  One inner list per cluster, in acquisition order.
    """
    if not rows:
        return []
    clusters: list[list[dict]] = [[rows[0]]]
    for r in rows[1:]:
        prev = clusters[-1][-1]
        avg_lat_rad = math.radians((r["lat"] + prev["lat"]) / 2.0)
        dlat = (r["lat"] - prev["lat"]) * 111_320.0
        dlon = (r["lon"] - prev["lon"]) * 111_320.0 * math.cos(avg_lat_rad)
        dist = math.sqrt(dlat * dlat + dlon * dlon)
        if dist > gap_threshold_m:
            clusters.append([r])
        else:
            clusters[-1].append(r)
    return clusters
