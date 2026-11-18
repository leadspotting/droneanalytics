from __future__ import annotations

from fastapi import HTTPException
from footprint_calculator import FootprintCalculator
from models import DroneTelemetry
from models import GPSCoordinate
from typing import Optional
import csv
import io
import math
import numpy as np
from backend.config import DEFAULT_CSV, _DEFAULT_CAMERA
from backend.state import _missions


def _parse_csv(content: str) -> dict[str, list[dict]]:
    """Parse CSV text, group rows by Mission_ID."""
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")), strict=True)
    required = {"Mission_ID", "Latitude", "Longitude", "Altitude_m"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("CSV must contain Mission_ID, Latitude, Longitude and Altitude_m.")
    missions: dict[str, list[dict]] = {}
    for line_number, row in enumerate(reader, start=2):
        mid = (row.get("Mission_ID") or "").strip()
        try:
            lat, lon, alt = (float(row[name]) for name in ("Latitude", "Longitude", "Altitude_m"))
            heading = float(row.get("Heading", row.get("heading", 0)) or 0)
        except (TypeError, ValueError):
            raise ValueError(f"CSV row {line_number} has missing or invalid telemetry.")
        if (not mid or None in row or not all(math.isfinite(v) for v in (lat, lon, alt, heading))
                or not -90 < lat < 90 or not -180 <= lon <= 180):
            raise ValueError(f"CSV row {line_number} has invalid mission or GPS values.")
        if mid not in missions:
            missions[mid] = []
        missions[mid].append(row)
    if not missions:
        raise ValueError("CSV contains no telemetry rows.")
    return missions


def _load_default() -> None:
    """Load the bundled CSV on startup if present."""
    if DEFAULT_CSV.exists() and not _missions:
        _missions.update(_parse_csv(DEFAULT_CSV.read_text(encoding="utf-8")))


def _row_to_footprint(row: dict) -> Optional[dict]:
    """Convert one telemetry CSV row into a footprint dict (4 GPS corners)."""
    try:
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        alt = float(row["Altitude_m"])
        heading = float(row.get("Heading", row.get("heading", 0.0)) or 0.0)
        if alt <= 0:
            alt = 1.0

        telemetry = DroneTelemetry(
            position=GPSCoordinate(lat=lat, lon=lon, alt=alt),
            heading=heading,
            camera=_DEFAULT_CAMERA,
        )
        result = FootprintCalculator(telemetry).compute()
        return {
            "nadir":  {"lat": lat, "lon": lon},
            "corners": [
                {"lat": c.lat, "lon": c.lon} for c in result.corners
            ],
            "width_m":  round(result.ground_width_m, 1),
            "height_m": round(result.ground_height_m, 1),
            "altitude_m": alt,
            "timestamp":  row.get("Timestamp", ""),
            "mission_type": row.get("Mission_Type", ""),
            "battery_pct":  row.get("Battery_pct", ""),
            "signal_pct":   row.get("Signal_pct", ""),
            "weather":      row.get("Weather_Condition", ""),
            "wind_m_s":     row.get("Wind_Speed_m_s", ""),
            "gps_tampering": row.get("GPS_Tampering", "0"),
            "data_leak":     row.get("Data_Leak_Detected", "0"),
            "memory_integrity": row.get("Memory_Integrity", "OK"),
            "drone_id":     row.get("Drone_ID", ""),
        }
    except Exception:
        return None


def _sanitize_for_json(obj):
    """Recursively sanitize numpy arrays and non-JSON-serializable objects for JSON output."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    else:
        return obj


def _mosaic_projection_data(mission_id: str, image_width_px: int) -> tuple:
    """
    Reconstruct mosaic bounding box and canvas dimensions from mission footprints.

    Raises HTTPException 404 / 422 as appropriate.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max, gsd_m, W_canvas, H_canvas)
    """
    from footprint_calculator import FootprintResult

    rows = _missions.get(mission_id)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"Mission '{mission_id}' not found.")

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

    if image_width_px <= 0:
        raise HTTPException(
            status_code=400,
            detail="image_width_px must be a positive integer.",
        )

    if not footprints:
        raise HTTPException(
            status_code=422,
            detail="No valid footprints in mission — cannot compute mosaic bounds.",
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


def _build_projection_mosaic(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    gsd_m: float,
    width_px: int,
    height_px: int,
):
    import numpy as np
    from drone_oversight import MosaicResult

    img_stub = np.broadcast_to(np.zeros(3, dtype=np.uint8), (height_px, width_px, 3))
    return MosaicResult(
        image=img_stub,
        bounding_box=[
            GPSCoordinate(lat=lat_max, lon=lon_min),
            GPSCoordinate(lat=lat_max, lon=lon_max),
            GPSCoordinate(lat=lat_min, lon=lon_max),
            GPSCoordinate(lat=lat_min, lon=lon_min),
        ],
        gsd_m=gsd_m,
    )
