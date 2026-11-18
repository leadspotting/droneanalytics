from __future__ import annotations

from drone_catalog import BUILTIN_DRONES
from drone_catalog import DroneSpec
from fastapi import HTTPException
from backend.state import _custom_drones


def _lookup_drone_spec(drone_id: str) -> DroneSpec:
    builtin = next((drone for drone in BUILTIN_DRONES if drone.id == drone_id), None)
    if builtin is not None:
        return builtin

    custom = next((drone for drone in _custom_drones if drone["id"] == drone_id), None)
    if custom is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown drone_id '{drone_id}'. Use GET /drones to list available drones.",
        )

    return DroneSpec(
        id=custom["id"],
        series=custom.get("series", "Custom"),
        model=custom["model"],
        cruise_speed_kmh=custom["cruise_speed_kmh"],
        max_speed_kmh=custom["max_speed_kmh"],
        max_flight_time_min=custom["max_flight_time_min"],
    )
