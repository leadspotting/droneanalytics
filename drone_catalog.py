"""
drone_catalog.py
================
Built-in DJI drone catalog and feasibility helpers for the mission planner.

Battery interpretation: when a manufacturer spec lists a range, the maximum
value is used for max_flight_time_min.  Safe flight time is then computed as
80% of that value (conservative field-safe estimate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

_SAFE_FACTOR = 0.8


@dataclass
class DroneSpec:
    """Specification for one drone model."""
    id: str                    # URL-safe slug, unique across all drones
    series: str                # e.g. "Mini", "Mavic", "Air", "Enterprise", "FPV"
    model: str                 # Display name, e.g. "DJI Mini 5 Pro"
    cruise_speed_kmh: float    # Typical cruise speed in km/h
    max_speed_kmh: float       # Maximum recorded speed in km/h
    max_flight_time_min: float # Manufacturer max battery time in minutes

    @property
    def safe_flight_time_min(self) -> float:
        """Conservative safe flight time = manufacturer max × 0.8."""
        return self.max_flight_time_min * _SAFE_FACTOR

    def to_dict(self) -> dict:
        return {
            "id":                   self.id,
            "series":               self.series,
            "model":                self.model,
            "cruise_speed_kmh":     self.cruise_speed_kmh,
            "max_speed_kmh":        self.max_speed_kmh,
            "max_flight_time_min":  self.max_flight_time_min,
            "safe_flight_time_min": round(self.safe_flight_time_min, 1),
            "custom":               False,
        }


# ---------------------------------------------------------------------------
# Built-in DJI catalog
# ---------------------------------------------------------------------------

BUILTIN_DRONES: List[DroneSpec] = [
    # ── Mini series ─────────────────────────────────────────────────────────
    DroneSpec("mini-5-pro",  "Mini", "DJI Mini 5 Pro",  43.0,  68.0, 52.0),
    DroneSpec("mini-4-pro",  "Mini", "DJI Mini 4 Pro",  43.0,  57.0, 45.0),
    DroneSpec("mini-3-pro",  "Mini", "DJI Mini 3 Pro",  36.0,  57.0, 51.0),
    DroneSpec("mini-2-se",   "Mini", "DJI Mini 2 SE",   36.0,  57.0, 31.0),

    # ── Mavic series ────────────────────────────────────────────────────────
    DroneSpec("mavic-4-pro",      "Mavic", "DJI Mavic 4 Pro",      64.0,  90.0, 51.0),
    DroneSpec("mavic-3-pro",      "Mavic", "DJI Mavic 3 Pro",      54.0,  75.0, 43.0),
    DroneSpec("mavic-3-classic",  "Mavic", "DJI Mavic 3 Classic",  54.0,  75.0, 46.0),
    DroneSpec("mavic-2-pro",      "Mavic", "DJI Mavic 2 Pro",      50.0,  72.0, 31.0),

    # ── Air series ──────────────────────────────────────────────────────────
    DroneSpec("air-3s",  "Air", "DJI Air 3S",  43.0, 75.0, 45.0),
    DroneSpec("air-3",   "Air", "DJI Air 3",   43.0, 75.0, 46.0),
    DroneSpec("air-2s",  "Air", "DJI Air 2S",  54.0, 68.0, 31.0),

    # ── Enterprise / production ─────────────────────────────────────────────
    DroneSpec("inspire-3",          "Enterprise", "DJI Inspire 3",          54.0,  94.0, 28.0),
    DroneSpec("matrice-350-rtk",    "Enterprise", "DJI Matrice 350 RTK",    54.0,  83.0, 55.0),
    DroneSpec("mavic-3-enterprise", "Enterprise", "DJI Mavic 3 Enterprise", 54.0,  75.0, 45.0),

    # ── FPV ─────────────────────────────────────────────────────────────────
    DroneSpec("avata-2", "FPV", "DJI Avata 2", 28.0,  97.0, 23.0),
    DroneSpec("dji-fpv", "FPV", "DJI FPV",     54.0, 140.0, 20.0),
]


# ---------------------------------------------------------------------------
# Feasibility helper
# ---------------------------------------------------------------------------

def check_feasibility(drone: DroneSpec, route_distance_m: float) -> dict:
    """
    Determine whether a planned route is feasible for the given drone.

    Parameters
    ----------
    drone            : DroneSpec
    route_distance_m : float   Total route length in metres.

    Returns
    -------
    dict with keys:
        feasible            bool
        estimated_time_min  float  Estimated flight time at cruise speed
        safe_flight_time_min float  Drone's safe battery limit (max × 0.8)
        distance_km         float  Route length in km
        battery_usage_pct   float  estimated / safe × 100 (capped at 200)
        reason              str    Empty when feasible; explanation when not
    """
    if route_distance_m <= 0.0:
        return {
            "feasible":             True,
            "estimated_time_min":   0.0,
            "safe_flight_time_min": round(drone.safe_flight_time_min, 1),
            "distance_km":          0.0,
            "battery_usage_pct":    0.0,
            "reason":               "",
        }

    cruise_speed_ms = drone.cruise_speed_kmh / 3.6
    estimated_time_min = (route_distance_m / cruise_speed_ms) / 60.0
    safe_time = drone.safe_flight_time_min
    feasible = estimated_time_min <= safe_time
    battery_pct = min(estimated_time_min / safe_time * 100.0, 200.0) if safe_time > 0 else 200.0

    reason = (
        ""
        if feasible
        else (
            f"Mission needs ~{estimated_time_min:.1f} min but safe limit for "
            f"{drone.model} is {safe_time:.1f} min "
            f"({drone.max_flight_time_min:.0f} min max × 0.8)"
        )
    )

    return {
        "feasible":             feasible,
        "estimated_time_min":   round(estimated_time_min, 1),
        "safe_flight_time_min": round(safe_time, 1),
        "distance_km":          round(route_distance_m / 1000.0, 2),
        "battery_usage_pct":    round(battery_pct, 1),
        "reason":               reason,
    }
