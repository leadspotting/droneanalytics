"""
srt_parser.py  —  DJI SRT telemetry file parser
=================================================
Parses DJI subtitle-format (.SRT) telemetry files into the row-dict
format used by the Drone Oversight mission store (api._missions).

Row shape produced (all values are strings, like CSV rows):
    Mission_ID, Drone_ID, Timestamp, Latitude, Longitude, Altitude_m,
    Heading, GPS_Tampering, Data_Leak_Detected, Memory_Integrity

Filtering rules:
- Frames where both latitude == 0 and longitude == 0 are dropped (pre-flight).

Heading derivation:
- First valid frame: heading = 0.0.
- Subsequent frames: compass bearing from previous to current position.
- Stationary frames (identical position): heading unchanged.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-field regex patterns  (each field matched independently for robustness)
# ---------------------------------------------------------------------------

_TS_RE        = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')
_LAT_RE       = re.compile(r'\[latitude:\s*([+-]?[\d.]+)\]')
_LON_RE       = re.compile(r'\[longitude:\s*([+-]?[\d.]+)\]')
_REL_ALT_RE   = re.compile(r'\[rel_alt:\s*([+-]?[\d.]+)')
_ABS_ALT_RE   = re.compile(r'abs_alt:\s*([+-]?[\d.]+)\]')
_FOCAL_LEN_RE = re.compile(r'\[focal_len:\s*([\d.]+)\]')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compass bearing (0° = North, clockwise) from (lat1, lon1) to (lat2, lon2).
    Returns a value in [0, 360).
    """
    avg_lat_rad = math.radians((lat1 + lat2) / 2.0)
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * math.cos(avg_lat_rad)
    return math.degrees(math.atan2(dlon, dlat)) % 360.0


def _parse_block(block_text: str) -> dict | None:
    """
    Extract telemetry fields from one SRT subtitle block.

    Returns a raw dict with float/str values, or None if the block is
    malformed, missing required GPS fields, or has zero lat/lon.
    """
    lat_m = _LAT_RE.search(block_text)
    lon_m = _LON_RE.search(block_text)
    rel_m = _REL_ALT_RE.search(block_text)
    if not (lat_m and lon_m and rel_m):
        return None  # incomplete / malformed block

    try:
        lat = float(lat_m.group(1))
        lon = float(lon_m.group(1))
        rel_alt = float(rel_m.group(1))
    except ValueError:
        return None

    # Drop pre-flight / parked frames
    if lat == 0.0 and lon == 0.0:
        return None

    abs_m = _ABS_ALT_RE.search(block_text)
    focal_m = _FOCAL_LEN_RE.search(block_text)
    ts_m = _TS_RE.search(block_text)

    return {
        "lat":       lat,
        "lon":       lon,
        "rel_alt":   rel_alt,
        "abs_alt":   float(abs_m.group(1)) if abs_m else None,
        "focal_len": float(focal_m.group(1)) if focal_m else None,
        "timestamp": ts_m.group(1) if ts_m else "",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_srt(text: str, mission_id: str = "SRT_MISSION") -> dict[str, list[dict]]:
    """
    Parse DJI SRT subtitle text into a missions dict.

    The returned structure is directly compatible with api._missions:
        {mission_id: [row_dict, ...]}

    Frames where latitude == 0 and longitude == 0 are filtered out.
    Heading is derived from consecutive valid positions; first frame
    defaults to 0.0°, stationary frames keep the previous heading.

    Parameters
    ----------
    text       : str   Raw SRT file content.
    mission_id : str   Key to use in the returned dict.

    Returns
    -------
    dict[str, list[dict]]   {mission_id: [row, ...]}
    """
    # SRT blocks are separated by blank lines
    blocks = re.split(r'\n\s*\n', text.strip())

    parsed: list[dict] = []
    for block in blocks:
        result = _parse_block(block)
        if result is not None:
            parsed.append(result)

    # Derive headings from consecutive valid positions
    rows: list[dict] = []
    prev_lat = prev_lon = None
    prev_heading = 0.0

    for p in parsed:
        lat, lon = p["lat"], p["lon"]
        if prev_lat is not None:
            dlat = lat - prev_lat
            dlon = lon - prev_lon
            if dlat != 0.0 or dlon != 0.0:
                prev_heading = _bearing(prev_lat, prev_lon, lat, lon)
        # When stationary, prev_heading is unchanged

        rows.append({
            "Mission_ID":         mission_id,
            "Drone_ID":           "DJI",
            "Timestamp":          p["timestamp"],
            "Latitude":           str(lat),
            "Longitude":          str(lon),
            "Altitude_m":         str(p["rel_alt"]),
            "Heading":            str(prev_heading),
            "GPS_Tampering":      "0",
            "Data_Leak_Detected": "0",
            "Memory_Integrity":   "OK",
        })
        prev_lat, prev_lon = lat, lon

    return {mission_id: rows}


def parse_srt_file(path: str | Path) -> dict[str, list[dict]]:
    """
    Parse a DJI SRT file. Uses the filename stem (uppercased) as mission_id.

    Parameters
    ----------
    path : str or Path   Path to the .srt file.

    Returns
    -------
    dict[str, list[dict]]
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_srt(text, mission_id=path.stem.upper())
