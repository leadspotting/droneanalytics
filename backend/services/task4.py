from __future__ import annotations

from fastapi import HTTPException
from footprint_calculator import FootprintCalculator
from models import CameraConfig
from models import DroneTelemetry
from models import GPSCoordinate
from models import GimbalMode
from pathlib import Path
import math
import re
from backend.config import TASK4_ORTHO_TIF, TASK4_ORTHO_XML, TASK4_UPLOAD_JPG, TASK4_UPLOAD_TIF


def _task4_reference_available() -> bool:
    return TASK4_UPLOAD_TIF.exists() or TASK4_UPLOAD_JPG.exists() or (TASK4_ORTHO_TIF.exists() and TASK4_ORTHO_XML.exists())


_WGS84_A = 6378137.0


_WGS84_F = 1 / 298.257223563


_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)


_WGS84_EP2 = _WGS84_E2 / (1 - _WGS84_E2)


_UTM_K0 = 0.9996


def _utm_central_meridian_rad(zone: int) -> float:
    return math.radians(zone * 6 - 183)


def _utm_to_latlon(easting_m: float, northing_m: float, zone: int, northern: bool = True) -> tuple[float, float]:
    x = easting_m - 500000.0
    y = northing_m if northern else northing_m - 10_000_000.0
    m = y / _UTM_K0
    mu = m / (
        _WGS84_A
        * (1 - _WGS84_E2 / 4 - 3 * _WGS84_E2 ** 2 / 64 - 5 * _WGS84_E2 ** 3 / 256)
    )

    e1 = (1 - math.sqrt(1 - _WGS84_E2)) / (1 + math.sqrt(1 - _WGS84_E2))
    j1 = 3 * e1 / 2 - 27 * e1 ** 3 / 32
    j2 = 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32
    j3 = 151 * e1 ** 3 / 96
    j4 = 1097 * e1 ** 4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = _WGS84_EP2 * cos_fp ** 2
    t1 = tan_fp ** 2
    n1 = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_fp ** 2)
    r1 = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * sin_fp ** 2) ** 1.5
    d = x / (n1 * _UTM_K0)

    lat = fp - (n1 * tan_fp / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _WGS84_EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _WGS84_EP2 - 3 * c1 ** 2) * d ** 6 / 720
    )

    lon = _utm_central_meridian_rad(zone) + (
        d
        - (1 + 2 * t1 + c1) * d ** 3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _WGS84_EP2 + 24 * t1 ** 2) * d ** 5 / 120
    ) / cos_fp

    return math.degrees(lat), math.degrees(lon)


def _infer_utm_zone(ascii_tag: str | None, geo_key_directory: tuple | None) -> tuple[int, bool]:
    if ascii_tag:
        match = re.search(r"UTM zone\s+(\d{1,2})([NS])", ascii_tag, re.IGNORECASE)
        if match:
            return int(match.group(1)), match.group(2).upper() == "N"

    if geo_key_directory:
        values = list(geo_key_directory)
        for index in range(4, len(values), 4):
            key_id, _, _, value = values[index:index + 4]
            if key_id == 3072:
                epsg = int(value)
                if 32601 <= epsg <= 32660:
                    return epsg - 32600, True
                if 32701 <= epsg <= 32760:
                    return epsg - 32700, False
                if 26901 <= epsg <= 26960:
                    return epsg - 26900, True

    raise RuntimeError("Unable to infer UTM zone from GeoTIFF metadata.")


def _load_task4_geotiff_metadata(tif_path: Path) -> dict:
    from PIL import Image as PILImage

    PILImage.MAX_IMAGE_PIXELS = None
    with PILImage.open(tif_path) as img:
        scale = img.tag_v2.get(33550)
        tie = img.tag_v2.get(33922)
        if scale is None or tie is None:
            raise RuntimeError("GeoTIFF missing georeferencing tags 33550/33922.")

        width_px, height_px = img.size
        scale_x = float(scale[0])
        scale_y = float(scale[1])
        origin_x = float(tie[3])
        origin_y = float(tie[4])
        ascii_tag = img.tag_v2.get(34737)
        geo_key_directory = img.tag_v2.get(34735)

        if abs(origin_x) <= 180 and abs(origin_y) <= 90:
            corners = [
                (origin_y, origin_x),
                (origin_y, origin_x + (width_px - 1) * scale_x),
                (origin_y - (height_px - 1) * scale_y, origin_x + (width_px - 1) * scale_x),
                (origin_y - (height_px - 1) * scale_y, origin_x),
            ]
        else:
            zone, northern = _infer_utm_zone(str(ascii_tag) if ascii_tag else None, geo_key_directory)
            corners = [
                _utm_to_latlon(origin_x, origin_y, zone, northern),
                _utm_to_latlon(origin_x + (width_px - 1) * scale_x, origin_y, zone, northern),
                _utm_to_latlon(origin_x + (width_px - 1) * scale_x, origin_y - (height_px - 1) * scale_y, zone, northern),
                _utm_to_latlon(origin_x, origin_y - (height_px - 1) * scale_y, zone, northern),
            ]

    lats = [lat for lat, _ in corners]
    lons = [lon for _, lon in corners]
    return {
        "dataset": "task4_reference",
        "reference_type": "geotiff",
        "source": "uploaded" if tif_path == TASK4_UPLOAD_TIF else "bundled",
        "filename": tif_path.name,
        "width_px": width_px,
        "height_px": height_px,
        "gsd_m": round(scale_x, 6),
        "bounding_box": {
            "west": min(lons),
            "east": max(lons),
            "north": max(lats),
            "south": min(lats),
        },
        "preview_url": "/task4/reference/preview?scale=0.02",
    }


def _load_task4_jpg_metadata(jpg_path: Path) -> dict:
    from PIL import Image as PILImage
    from exif_loader import (
        RICOH_GR_II_FOV_H,
        RICOH_GR_II_FOV_V,
        SEKI_AGL_M,
        _extract_gps,
    )

    with PILImage.open(jpg_path) as img:
        exif_data = img._getexif()
        if not exif_data:
            raise RuntimeError("JPG is missing EXIF metadata.")
        gps = _extract_gps(exif_data)
        if gps is None:
            raise RuntimeError("JPG is missing usable GPS EXIF metadata.")

        width_px, height_px = img.size

    telemetry = DroneTelemetry(
        position=GPSCoordinate(lat=gps["lat"], lon=gps["lon"], alt=SEKI_AGL_M),
        heading=0.0,
        camera=CameraConfig(
            fov_h=RICOH_GR_II_FOV_H,
            fov_v=RICOH_GR_II_FOV_V,
            gimbal_mode=GimbalMode.NADIR,
        ),
    )
    result = FootprintCalculator(telemetry).compute()
    corners = [{"lat": c.lat, "lon": c.lon} for c in result.corners]
    lats = [corner["lat"] for corner in corners]
    lons = [corner["lon"] for corner in corners]

    return {
        "dataset": "task4_reference",
        "reference_type": "jpg_exif",
        "source": "uploaded",
        "filename": jpg_path.name,
        "width_px": width_px,
        "height_px": height_px,
        "gsd_m": round(result.ground_width_m / max(width_px, 1), 6),
        "bounding_box": {
            "west": min(lons),
            "east": max(lons),
            "north": max(lats),
            "south": min(lats),
        },
        "nadir": {
            "lat": gps["lat"],
            "lon": gps["lon"],
        },
        "corners": corners,
        "preview_url": "/task4/reference/preview?scale=1.0",
    }


def _get_task4_reference_metadata() -> dict:
    if TASK4_UPLOAD_TIF.exists():
        return _load_task4_geotiff_metadata(TASK4_UPLOAD_TIF)
    if TASK4_UPLOAD_JPG.exists():
        return _load_task4_jpg_metadata(TASK4_UPLOAD_JPG)

    if not (TASK4_ORTHO_TIF.exists() and TASK4_ORTHO_XML.exists()):
        raise HTTPException(status_code=404, detail="Task 4 reference data not found.")

    import xml.etree.ElementTree as ET
    from PIL import Image as PILImage

    PILImage.MAX_IMAGE_PIXELS = None
    root = ET.parse(TASK4_ORTHO_XML).getroot()
    bounding = root.find(".//bounding")
    if bounding is None:
        raise HTTPException(status_code=500, detail="Task 4 metadata XML missing bounding box.")

    with PILImage.open(TASK4_ORTHO_TIF) as img:
        scale = img.tag_v2.get(33550)
        return {
            "dataset": "task4_reference",
            "reference_type": "geotiff",
            "source": "bundled",
            "filename": TASK4_ORTHO_TIF.name,
            "width_px": img.size[0],
            "height_px": img.size[1],
            "gsd_m": round(float(scale[0]), 6) if scale else None,
            "bounding_box": {
                "west": float(bounding.find("westbc").text),
                "east": float(bounding.find("eastbc").text),
                "north": float(bounding.find("northbc").text),
                "south": float(bounding.find("southbc").text),
            },
            "preview_url": "/task4/reference/preview?scale=0.02",
        }


def _get_task4_reference_tif_path() -> Path:
    if TASK4_UPLOAD_TIF.exists():
        return TASK4_UPLOAD_TIF
    if TASK4_UPLOAD_JPG.exists():
        return TASK4_UPLOAD_JPG
    return TASK4_ORTHO_TIF
