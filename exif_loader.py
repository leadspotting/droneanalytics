"""
exif_loader.py  —  EXIF-based image dataset loader for Task 3
=============================================================
Reads geotagged JPGs from a directory, extracts GPS metadata from EXIF,
and derives per-image headings from consecutive positions.

Produces a list of image-row dicts compatible with the _image_missions store
in api.py.  Completely independent of the CSV/SRT mission pipeline.

Dataset notes (SEKI SouthFork, 2024-04-30):
  Camera  : Ricoh GR II (APS-C 23.7x15.7 mm sensor, 18.3 mm focal length)
  Altitude: EXIF stores WGS84 ellipsoid height (~1465 m at SEKI), NOT AGL.
            agl_m is therefore supplied as a dataset-level approximation (110 m)
            derived from the survey metadata XML, not from per-frame EXIF.
  Heading : GPSImgDirection is absent from these images; heading is inferred
            from the compass bearing between consecutive image positions after
            sorting by filename (filenames encode acquisition UTC timestamps).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ricoh GR II constants
# ---------------------------------------------------------------------------
# Sensor: APS-C  23.7 x 15.7 mm  |  focal length: 18.3 mm
# FOV_h = 2 * atan(23.7 / (2 * 18.3)) = 65.8 deg
# FOV_v = 2 * atan(15.7 / (2 * 18.3)) = 46.4 deg
# These match the dataset's documented 2.9 cm/px GSD at 110 m AGL.
RICOH_GR_II_FOV_H: float = 65.8
RICOH_GR_II_FOV_V: float = 46.4
RICOH_GR_II_W_PX: int = 4928
RICOH_GR_II_H_PX: int = 3264

# Default AGL for the SEKI SouthFork dataset.
# Source: survey metadata XML ("approximate altitude of 110 meters above ground level").
# This is a dataset-level approximation, not a per-frame measurement.
SEKI_AGL_M: float = 110.0


# ---------------------------------------------------------------------------
# Internal helpers (module-level so they are unit-testable)
# ---------------------------------------------------------------------------

def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """
    Convert an EXIF DMS tuple (degrees, minutes, seconds) to decimal degrees.

    Parameters
    ----------
    dms : tuple   Degrees, minutes, seconds (floats).
    ref : str     Hemisphere reference ('N', 'S', 'E', 'W').

    Returns
    -------
    float   Decimal degrees (negative for S / W).
    """
    d, m, s = dms
    decimal = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compass bearing (0 deg = North, clockwise) from (lat1, lon1) to (lat2, lon2).

    Uses a flat-earth approximation suitable for short drone-flight distances.
    Returns a value in [0, 360).
    """
    avg_lat_rad = math.radians((lat1 + lat2) / 2.0)
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * math.cos(avg_lat_rad)
    return math.degrees(math.atan2(dlon, dlat)) % 360.0


def _extract_gps(exif_data: dict[int, Any]) -> dict | None:
    """
    Pull GPS fields out of a raw EXIF integer-keyed dict.

    Returns a plain dict with parsed lat/lon/alt/timestamp, or None if
    required GPS fields are missing, malformed, or both lat and lon are zero.
    """
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
    except ImportError:  # pragma: no cover
        raise RuntimeError("Pillow is required: pip install Pillow")

    gps_info: dict | None = None
    for tag_id, val in exif_data.items():
        if TAGS.get(tag_id) == "GPSInfo":
            gps_info = {GPSTAGS.get(k, k): v for k, v in val.items()}
            break

    if gps_info is None:
        return None

    lat_dms = gps_info.get("GPSLatitude")
    lat_ref = gps_info.get("GPSLatitudeRef", "N")
    lon_dms = gps_info.get("GPSLongitude")
    lon_ref = gps_info.get("GPSLongitudeRef", "E")
    alt_raw = gps_info.get("GPSAltitude")
    ts_time = gps_info.get("GPSTimeStamp")   # (h, m, s) as floats
    ts_date = gps_info.get("GPSDateStamp", "")  # "YYYY:MM:DD"

    if lat_dms is None or lon_dms is None:
        return None

    try:
        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)
    except (TypeError, ValueError):
        return None

    if lat == 0.0 and lon == 0.0:
        return None

    timestamp = ""
    if ts_date and ts_time:
        try:
            h, m, s = ts_time
            timestamp = f"{ts_date} {int(h):02d}:{int(m):02d}:{int(s):02d}Z"
        except (TypeError, ValueError):
            pass

    return {
        "lat": lat,
        "lon": lon,
        "alt_ellipsoid_m": float(alt_raw) if alt_raw is not None else None,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_image_dir(
    directory: Path,
    agl_m: float = SEKI_AGL_M,
) -> list[dict]:
    """
    Load all JPGs from *directory*, extract EXIF GPS, and derive headings.

    Images are processed in **filename order** (SEKI filenames embed UTC
    timestamps, so alphabetical sort == chronological acquisition order).

    Parameters
    ----------
    directory : Path    Directory containing geotagged JPG files.
    agl_m     : float   Above-ground-level altitude applied to all images.
                        This is a dataset-level approximation; EXIF stores
                        WGS84 ellipsoid height, not AGL.

    Returns
    -------
    list[dict]  One dict per valid image, containing:
                  path, lat, lon, agl_m, alt_ellipsoid_m, timestamp,
                  heading, width_px, height_px, fov_h, fov_v

    Raises
    ------
    FileNotFoundError   If *directory* does not exist.
    RuntimeError        If Pillow is not installed.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    try:
        from PIL import Image as PILImage
    except ImportError:  # pragma: no cover
        raise RuntimeError("Pillow is required for EXIF loading: pip install Pillow")

    # Sort by filename so heading derivation is deterministic (filenames = timestamps).
    # Use iterdir + lower-case suffix check to avoid duplicates on case-insensitive
    # filesystems (Windows glob("*.jpg") and glob("*.JPG") both match the same files).
    jpg_files = sorted(
        [p for p in directory.iterdir() if p.suffix.lower() == ".jpg"],
        key=lambda p: p.name,
    )

    # First pass: extract raw GPS from each file
    raw: list[dict] = []
    for path in jpg_files:
        try:
            img = PILImage.open(path)
            exif_data = img._getexif()
            if not exif_data:
                continue
            gps = _extract_gps(exif_data)
            if gps is None:
                continue
            raw.append({**gps, "path": str(path)})
        except Exception:
            continue

    # Second pass: derive headings from consecutive valid positions
    rows: list[dict] = []
    prev_lat: float | None = None
    prev_lon: float | None = None
    prev_heading: float = 0.0

    for r in raw:
        lat, lon = r["lat"], r["lon"]
        if prev_lat is not None and (lat != prev_lat or lon != prev_lon):
            prev_heading = _bearing(prev_lat, prev_lon, lat, lon)
        # Stationary images keep prev_heading unchanged

        rows.append({
            "path":            r["path"],
            "lat":             lat,
            "lon":             lon,
            "agl_m":           agl_m,
            "alt_ellipsoid_m": r["alt_ellipsoid_m"],
            "timestamp":       r["timestamp"],
            "heading":         prev_heading,
            "width_px":        RICOH_GR_II_W_PX,
            "height_px":       RICOH_GR_II_H_PX,
            "fov_h":           RICOH_GR_II_FOV_H,
            "fov_v":           RICOH_GR_II_FOV_V,
        })
        prev_lat, prev_lon = lat, lon

    return rows
