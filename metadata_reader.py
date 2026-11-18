"""
metadata_reader.py — DJI EXIF + XMP metadata extraction
========================================================
Reads DJI image metadata including GPS, gimbal angles, focal length, and dimensions.
Returns DJIImageMetadata dataclass for each image.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import warnings

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import xml.etree.ElementTree as ET

from models import DJIImageMetadata


def read_dji_metadata(image_path: str) -> DJIImageMetadata:
    """
    Extract DJI metadata from a single image file.

    Returns DJIImageMetadata with:
    - GPS coordinates (lat, lon, alt_abs) from EXIF
    - Image dimensions from EXIF
    - Focal length from EXIF
    - Gimbal angles and relative altitude from XMP (drone-dji: namespace)
    - Fallback values if XMP absent

    Raises ValueError if GPS data is invalid or missing.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Read EXIF ───────────────────────────────────────────────────────────
    img = Image.open(image_path)
    exif_data = img.getexif()

    # Extract GPS
    lat, lon, alt_abs = _extract_gps_from_exif(exif_data)

    # Extract focal length (in mm)
    focal_length_mm = _extract_focal_length(exif_data)

    # Extract image dimensions
    image_width_px, image_height_px = img.size

    # ── Read XMP for DJI-specific fields ─────────────────────────────────────
    xmp_str = _get_xmp_string(image_path)
    gimbal_yaw, gimbal_pitch, gimbal_roll, alt_rel, flight_yaw = _parse_dji_xmp(xmp_str)

    # Fallback: estimate alt_rel if not in XMP (assume modest AGL)
    if alt_rel is None:
        alt_rel = max(0.0, alt_abs - 1000.0)  # Assume takeoff at ~1000m elevation

    return DJIImageMetadata(
        path=str(path),
        lat=lat,
        lon=lon,
        alt_abs=alt_abs,
        alt_rel=alt_rel,
        gimbal_yaw=gimbal_yaw,
        gimbal_pitch=gimbal_pitch,
        gimbal_roll=gimbal_roll,
        flight_yaw=flight_yaw,
        focal_length_mm=focal_length_mm,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )


def read_dataset(directory: str) -> list[DJIImageMetadata]:
    """
    Load all DJI images from a directory.

    Returns list of DJIImageMetadata, sorted by filename (chronological for DJI).
    Raises ValueError if fewer than 3 images with valid GPS found.
    """
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Find all image files (case-insensitive)
    supported_formats = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    image_paths = sorted([
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in supported_formats
    ])

    if not image_paths:
        raise ValueError(f"No images found in {directory}")

    metas = []
    for img_path in image_paths:
        try:
            meta = read_dji_metadata(str(img_path))
            metas.append(meta)
        except (ValueError, FileNotFoundError, OSError) as e:
            warnings.warn(f"Skipping {img_path.name}: {e}", stacklevel=2)
            continue

    if len(metas) < 3:
        raise ValueError(
            f"Fewer than 3 images with valid metadata found in {directory}. "
            f"Got {len(metas)}."
        )

    # Sort by filename (chronological for DJI naming)
    metas.sort(key=lambda m: Path(m.path).name)

    return metas


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gps_from_exif(exif_data) -> tuple[float, float, float]:
    """
    Extract (lat, lon, alt_abs) from EXIF GPS tags.

    Returns tuple of (latitude, longitude, altitude_metres).
    Raises ValueError if GPS data is missing or invalid.
    """
    gps_ifd = exif_data.get_ifd(0x8825)  # GPS IFD pointer
    if not gps_ifd:
        raise ValueError("No GPS data in EXIF")

    # GPS tags: 1=N/S, 2=Lat, 3=E/W, 4=Lon, 6=Alt
    gps_lat_ref = gps_ifd.get(1)  # 'N' or 'S'
    gps_lat = gps_ifd.get(2)      # (degrees, minutes, seconds) tuples
    gps_lon_ref = gps_ifd.get(3)  # 'E' or 'W'
    gps_lon = gps_ifd.get(4)      # (degrees, minutes, seconds) tuples
    gps_alt = gps_ifd.get(6)      # altitude in metres

    if not all([gps_lat, gps_lon]):
        raise ValueError("Incomplete GPS data")

    # Convert DMS to decimal degrees
    lat = _dms_to_decimal(gps_lat)
    lon = _dms_to_decimal(gps_lon)

    # Apply hemisphere
    if gps_lat_ref and gps_lat_ref.upper() == 'S':
        lat = -lat
    if gps_lon_ref and gps_lon_ref.upper() == 'W':
        lon = -lon

    # Altitude
    alt_abs = float(gps_alt) if gps_alt else 0.0

    # Sanity check
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid GPS coordinates: ({lat}, {lon})")

    return lat, lon, alt_abs


def _dms_to_decimal(dms_tuple: tuple) -> float:
    """Convert (degrees, minutes, seconds) to decimal degrees."""
    d, m, s = dms_tuple
    decimal = float(d) + float(m) / 60.0 + float(s) / 3600.0
    return decimal


def _extract_focal_length(exif_data) -> float:
    """
    Extract focal length from EXIF (tag 37386 = FocalLength).

    Returns focal length in millimetres, or 24.0 (sensible default) if missing.
    """
    try:
        # Tag 37386 = FocalLength
        focal_length = exif_data.get(37386)
        if focal_length:
            # FocalLength is a rational (numerator, denominator)
            if isinstance(focal_length, tuple):
                return float(focal_length[0]) / float(focal_length[1])
            else:
                return float(focal_length)
    except (KeyError, TypeError, ZeroDivisionError):
        pass

    # Fallback: DJI Mavic 3 has ~162mm (full-frame equivalent ~24mm, but we want the actual optics)
    # For Ricoh GR II and typical DJI cameras, ~24mm equivalent is ~24mm actual focal length
    # Use 24mm as a reasonable default
    return 24.0


def _get_xmp_string(image_path: str) -> str:
    """
    Extract raw XMP XML string from image file.

    Uses PIL's info dict (XMP is sometimes stored there) or returns empty string
    if not found.
    """
    try:
        img = Image.open(image_path)
        # XMP is often in the 'xmp' key of img.info or in APP1 segment
        if "xmp" in img.info:
            return img.info["xmp"]
        # Try piexif if available
        try:
            import piexif
            exif_dict = piexif.load(image_path)
            if piexif.ImageIFD.InteroperabilityIFD in exif_dict:
                # XMP might be here
                pass
        except ImportError:
            pass
    except Exception:
        pass

    return ""


def _parse_dji_xmp(xmp_str: str) -> tuple[float, float, float, Optional[float], float]:
    """
    Parse DJI-specific fields from XMP namespace 'drone-dji:'.

    Looks for:
    - GimbalYaw, GimbalPitch, GimbalRoll (degrees)
    - RelativeAltitude (metres)
    - FlightYaw (degrees, flight heading)

    Returns (gimbal_yaw, gimbal_pitch, gimbal_roll, alt_rel, flight_yaw).
    Uses sensible fallbacks if XMP absent or unparseable.
    """
    gimbal_yaw = 0.0
    gimbal_pitch = -90.0      # Nadir (looking straight down)
    gimbal_roll = 0.0
    alt_rel = None
    flight_yaw = 0.0

    if not xmp_str:
        return gimbal_yaw, gimbal_pitch, gimbal_roll, alt_rel, flight_yaw

    try:
        # Parse XMP XML (namespace-aware)
        root = ET.fromstring(xmp_str)

        # DJI namespace: http://www.dji.com/drone/...
        ns = {
            'drone-dji': 'http://www.dji.com/drone/1.0/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        }

        # Try to find DJI elements (may be in any namespace variant)
        for elem in root.iter():
            tag = elem.tag
            text = elem.text or ""

            if "GimbalYaw" in tag:
                try:
                    gimbal_yaw = float(text)
                except ValueError:
                    pass
            elif "GimbalPitch" in tag:
                try:
                    gimbal_pitch = float(text)
                except ValueError:
                    pass
            elif "GimbalRoll" in tag:
                try:
                    gimbal_roll = float(text)
                except ValueError:
                    pass
            elif "RelativeAltitude" in tag:
                try:
                    alt_rel = float(text)
                except ValueError:
                    pass
            elif "FlightYaw" in tag or "Yaw" in tag:
                try:
                    flight_yaw = float(text)
                except ValueError:
                    pass
    except ET.ParseError:
        pass  # XMP unparseable; use fallbacks

    return gimbal_yaw, gimbal_pitch, gimbal_roll, alt_rel, flight_yaw
