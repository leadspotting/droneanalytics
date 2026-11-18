from __future__ import annotations

from coverage_heatmap import CoverageHeatmap
from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from models import GPSCoordinate
from pathlib import Path
import csv
import math
from backend.services.telemetry import _build_projection_mosaic, _mosaic_projection_data, _parse_csv, _row_to_footprint
from backend.config import MAX_CSV_UPLOAD_BYTES, MAX_HEATMAP_CELLS
from backend.state import _missions

router = APIRouter()


@router.get("/missions")
async def list_missions():
    """Return all mission IDs with basic metadata."""
    result = []
    for mid, rows in _missions.items():
        if not rows:
            continue
        first = rows[0]
        result.append({
            "mission_id":   mid,
            "drone_id":     first.get("Drone_ID", ""),
            "operator_id":  first.get("Operator_ID", ""),
            "from":         first.get("Takeoff_Location", ""),
            "to":           first.get("Landing_Location", ""),
            "waypoints":    len(rows),
            "start_time":   first.get("Timestamp", ""),
        })
    return result


@router.get("/mission/{mission_id}/footprints")
async def get_footprints(mission_id: str):
    """Task 1: return footprint polygons + metadata for every telemetry point."""
    rows = _missions.get(mission_id)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")

    footprints = [fp for row in rows if (fp := _row_to_footprint(row)) is not None]
    return {"mission_id": mission_id, "count": len(footprints), "footprints": footprints}


@router.get("/mission/{mission_id}/heatmap")
async def get_heatmap(
    mission_id: str,
    resolution_m: float = 200.0,
    drone_id: str | None = None,
):
    """Task 2: return the coverage heatmap as a list of coloured cells."""
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise HTTPException(status_code=400, detail="resolution_m must be finite and positive.")
    rows = _missions.get(mission_id)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")

    if not rows:
        raise HTTPException(status_code=400, detail="Mission has no telemetry rows.")

    # Build AOI bounding box from the telemetry path
    lats = [float(r["Latitude"]) for r in rows]
    lons = [float(r["Longitude"]) for r in rows]
    margin = 0.01   # ~1 km padding
    area = [
        GPSCoordinate(lat=min(lats) - margin, lon=min(lons) - margin),
        GPSCoordinate(lat=min(lats) - margin, lon=max(lons) + margin),
        GPSCoordinate(lat=max(lats) + margin, lon=max(lons) + margin),
        GPSCoordinate(lat=max(lats) + margin, lon=min(lons) - margin),
    ]

    filtered_rows = rows
    if drone_id:
        filtered_rows = [
            row for row in rows
            if (row.get("Drone_ID") or row.get("drone_id") or "").strip() == drone_id
        ]

    heatmap = CoverageHeatmap(area, grid_resolution_m=resolution_m)
    if heatmap._rows * heatmap._cols > MAX_HEATMAP_CELLS:
        raise HTTPException(
            status_code=400,
            detail=f"resolution_m too fine for this area ({heatmap._rows * heatmap._cols} cells, max {MAX_HEATMAP_CELLS}). Use a larger value.",
        )
    for row in filtered_rows:
        fp = _row_to_footprint(row)
        if fp is None:
            continue
        # Re-build a FootprintResult-compatible object from dict
        from footprint_calculator import FootprintResult
        corners = [GPSCoordinate(lat=c["lat"], lon=c["lon"]) for c in fp["corners"]]
        fr = FootprintResult(
            top_left=corners[0], top_right=corners[1],
            bottom_right=corners[2], bottom_left=corners[3],
            nadir=GPSCoordinate(lat=fp["nadir"]["lat"], lon=fp["nadir"]["lon"]),
            ground_width_m=fp["width_m"], ground_height_m=fp["height_m"],
        )
        heatmap.add_footprint(fr)

    grid = heatmap.get_grid()
    cells = []
    rows_idx, cols_idx = grid.nonzero()
    for r, c in zip(rows_idx.tolist(), cols_idx.tolist()):
        centre = heatmap.get_cell_center(int(r), int(c))
        cells.append({
            "lat": centre.lat,
            "lon": centre.lon,
            "count": int(grid[r, c]),
        })

    return {
        "mission_id": mission_id,
        "drone_filter_applied": drone_id or None,
        "coverage_pct": round(heatmap.coverage_percentage(), 1),
        "max_overlap":  heatmap.max_overlap(),
        "resolution_m": resolution_m,
        "cells": cells,
    }


@router.get("/mission/{mission_id}/mosaic")
async def get_mosaic_metadata(mission_id: str, image_width_px: int = 4000):
    """
    Task 3: Return mosaic canvas metadata derived from mission footprints.

    Because the in-memory mission store holds only telemetry rows (no image
    arrays), this endpoint computes the mosaic bounding box, GSD, and canvas
    dimensions that ImageStitcher.stitch() would produce — without running the
    actual stitch.  Use image_width_px to model your sensor resolution.
    """
    rows = _missions.get(mission_id)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")

    from footprint_calculator import FootprintResult
    footprints: list[FootprintResult] = []
    for row in rows:
        fp_dict = _row_to_footprint(row)
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
            detail="No valid footprints in mission — cannot compute mosaic bounds.",
        )

    # Canvas bounds: union of all footprint GPS corners
    all_lats = [c.lat for fp in footprints for c in fp.corners]
    all_lons = [c.lon for fp in footprints for c in fp.corners]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)

    # GSD: finest resolution across footprints (same logic as ImageStitcher.stitch)
    METRES_PER_LAT_DEG = 111_320.0
    avg_lat_rad = math.radians((lat_min + lat_max) / 2.0)
    gsd_m = min(fp.ground_width_m / max(image_width_px, 1) for fp in footprints)

    # Canvas pixel dimensions
    lat_span_m = (lat_max - lat_min) * METRES_PER_LAT_DEG
    lon_span_m = (lon_max - lon_min) * METRES_PER_LAT_DEG * math.cos(avg_lat_rad)
    W_canvas = max(1, round(lon_span_m / gsd_m))
    H_canvas = max(1, round(lat_span_m / gsd_m))

    return {
        "mission_id":       mission_id,
        "status":           "metadata_only",
        "message":          (
            "Mosaic metadata computed from telemetry footprints. "
            "No image arrays in mission store — upload actual drone images "
            "and call ImageStitcher.stitch() directly to produce a real mosaic."
        ),
        "footprint_count":  len(footprints),
        "bounding_box": {
            "TL": {"lat": lat_max, "lon": lon_min},
            "TR": {"lat": lat_max, "lon": lon_max},
            "BR": {"lat": lat_min, "lon": lon_max},
            "BL": {"lat": lat_min, "lon": lon_min},
        },
        "gsd_m":            round(gsd_m, 6),
        "canvas_width_px":  W_canvas,
        "canvas_height_px": H_canvas,
        "image_width_px":   image_width_px,
    }


@router.get("/mission/{mission_id}/project")
async def project_pixel(
    mission_id: str,
    x: int,
    y: int,
    image_width_px: int = 4000,
):
    """
    Task 4: Convert a mosaic pixel coordinate to a GPS position.

    x, y are 0-indexed pixel offsets into the mosaic canvas (0 = top-left).
    Use the same image_width_px as /mission/{id}/mosaic to stay consistent;
    the canvas_width_px and canvas_height_px returned here match that route.

    Returns 404 if the mission is missing, 422 if it has no valid footprints,
    and 400 if x or y falls outside the computed canvas bounds.
    """
    lat_min, lat_max, lon_min, lon_max, gsd_m, W_canvas, H_canvas = (
        _mosaic_projection_data(mission_id, image_width_px)
    )

    from drone_oversight import GPSProjector, MosaicResult

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


@router.get("/mission/{mission_id}/project/reverse")
async def project_gps_to_pixel(
    mission_id: str,
    lat: float,
    lon: float,
    image_width_px: int = 4000,
):
    """
    Task 4: Convert a GPS coordinate inside the mosaic bounding box to a pixel.
    """
    lat_min, lat_max, lon_min, lon_max, gsd_m, W_canvas, H_canvas = (
        _mosaic_projection_data(mission_id, image_width_px)
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


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV or SRT upload, parse it, and replace the in-memory dataset."""
    filename = (file.filename or "").lower()
    raw = await file.read(MAX_CSV_UPLOAD_BYTES + 1)
    if len(raw) > MAX_CSV_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")
    content = raw.decode("utf-8", errors="replace")
    try:
        if filename.endswith(".srt"):
            from srt_parser import parse_srt
            mission_id = Path(file.filename).stem.upper()
            parsed = parse_srt(content, mission_id=mission_id)
            if not any(parsed.values()):
                raise ValueError("SRT contains no valid GPS telemetry.")
        elif filename.endswith(".csv"):
            parsed = _parse_csv(content)
        else:
            raise HTTPException(status_code=400, detail="Only CSV or SRT files are accepted.")
    except (ValueError, csv.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _missions.clear()
    _missions.update(parsed)
    total = sum(len(v) for v in _missions.values())
    return {
        "status": "ok",
        "missions": list(_missions.keys()),
        "total_rows": total,
    }
