from __future__ import annotations

from drone_catalog import BUILTIN_DRONES
from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from backend.state import _custom_drones

router = APIRouter()


@router.get("/drones")
async def list_drones():
    """Return the full drone catalog (built-in + any custom drones added this session)."""
    return {"drones": [d.to_dict() for d in BUILTIN_DRONES] + _custom_drones}


class CustomDroneRequest(BaseModel):
    model: str
    series: str = "Custom"
    cruise_speed_kmh: float
    max_speed_kmh: float
    max_flight_time_min: float


@router.post("/drones")
async def add_custom_drone(req: CustomDroneRequest):
    """Add a custom drone to the in-session catalog."""
    if not req.model.strip():
        raise HTTPException(status_code=400, detail="model must not be empty.")
    if req.cruise_speed_kmh <= 0 or req.max_speed_kmh <= 0 or req.max_flight_time_min <= 0:
        raise HTTPException(status_code=400, detail="Speed and flight time values must be positive.")

    drone_id = "custom-" + req.model.strip().lower().replace(" ", "-")

    # Reject duplicates (check both built-in IDs and existing custom IDs)
    builtin_ids = {d.id for d in BUILTIN_DRONES}
    custom_ids  = {d["id"] for d in _custom_drones}
    if drone_id in builtin_ids or drone_id in custom_ids:
        raise HTTPException(status_code=409, detail=f"A drone with id '{drone_id}' already exists.")

    safe_time = round(req.max_flight_time_min * 0.8, 1)
    new_drone: dict = {
        "id":                   drone_id,
        "series":               req.series.strip() or "Custom",
        "model":                req.model.strip(),
        "cruise_speed_kmh":     req.cruise_speed_kmh,
        "max_speed_kmh":        req.max_speed_kmh,
        "max_flight_time_min":  req.max_flight_time_min,
        "safe_flight_time_min": safe_time,
        "custom":               True,
    }
    _custom_drones.append(new_drone)

    full_list = [d.to_dict() for d in BUILTIN_DRONES] + _custom_drones
    return {"drone": new_drone, "drones": full_list}
