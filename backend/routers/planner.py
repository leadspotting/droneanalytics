from __future__ import annotations

from drone_catalog import BUILTIN_DRONES
from drone_catalog import DroneSpec
from drone_catalog import check_feasibility
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from backend.config import MAX_PLAN_WAYPOINTS
from fastapi import HTTPException
from models import CameraConfig
from models import GPSCoordinate
from models import GimbalMode
from pydantic import BaseModel
from typing import Optional
import math
from backend.config import _DEFAULT_CAMERA
from backend.services.drones import _lookup_drone_spec
from backend.state import _custom_drones

router = APIRouter()


class Point(BaseModel):
    lat: float
    lon: float


class PlanRequest(BaseModel):
    area: list[Point]
    altitude_m: float = 50.0
    overlap_h: float = 0.5
    overlap_v: float = 0.5
    # ── Optional constraint inputs ────────────────────────────────────────────
    # Providing either drone_id (catalog lookup) or both cruise_speed_kmh +
    # safe_flight_time_min (inline) enables server-side feasibility checking.
    # Omitting all three preserves the original backward-compatible response.
    drone_id: Optional[str] = None
    cruise_speed_kmh: Optional[float] = None
    safe_flight_time_min: Optional[float] = None


@router.post("/plan_mission")
async def plan_mission(req: PlanRequest):
    """Task 6: Generate optimal waypoints to cover the area polygon.

    Constraint-aware mode (optional):
      - Pass ``drone_id`` to look up a catalog drone and get a server-side
        feasibility verdict.
      - Pass ``cruise_speed_kmh`` + ``safe_flight_time_min`` for an inline
        check without needing a catalog entry.
      - Omitting all constraint fields returns the same response shape as
        before (backward-compatible); no ``feasibility`` field is added.

    The route is always returned even when infeasible — callers inspect
    ``feasibility.feasible`` to decide whether to proceed.
    """
    from mission_planner import MissionPlanner

    if len(req.area) < 3:
        raise HTTPException(status_code=400, detail="Area must have at least 3 points.")

    polygon = [GPSCoordinate(lat=p.lat, lon=p.lon) for p in req.area]

    if not all(math.isfinite(value) for value in (req.altitude_m, req.overlap_h, req.overlap_v)):
        raise HTTPException(status_code=400, detail="Flight parameters must be finite.")
    if any(not math.isfinite(p.lat) or not math.isfinite(p.lon)
           or not -90 < p.lat < 90 or not -180 <= p.lon <= 180 for p in req.area):
        raise HTTPException(status_code=400, detail="Area contains invalid GPS coordinates.")
    try:
        planner = MissionPlanner(
            area_polygon=polygon,
            altitude_m=req.altitude_m,
            camera_config=_DEFAULT_CAMERA,
            overlap_h=req.overlap_h,
            overlap_v=req.overlap_v
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    estimated = planner.estimate_waypoint_count()
    if estimated > MAX_PLAN_WAYPOINTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Area too large for these settings (~{estimated:,} waypoints, max {MAX_PLAN_WAYPOINTS:,}). "
                "Draw a smaller area, raise the altitude or lower the overlap."
            ),
        )
    # CPU-bound: run off the event loop so the rest of the API stays responsive.
    waypoints = await run_in_threadpool(planner.plan)
    route_distance_m = planner.compute_route_distance(waypoints)
    flight_time_sec = route_distance_m / 10.0  # legacy: assumes 10 m/s

    # ── Constraint-aware feasibility (optional) ───────────────────────────────
    feasibility_result: Optional[dict] = None

    if req.drone_id is not None:
        # Path 1: catalog lookup by drone_id (built-ins + custom drones added
        # to this session via POST /drones).
        drone: Optional[DroneSpec] = next(
            (d for d in BUILTIN_DRONES if d.id == req.drone_id), None
        )
        if drone is None:
            # Try custom drones (stored as plain dicts, not DroneSpec)
            custom = next((d for d in _custom_drones if d["id"] == req.drone_id), None)
            if custom is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown drone_id '{req.drone_id}'. "
                        "Use GET /drones to list available drone IDs."
                    ),
                )
            drone = DroneSpec(
                id=custom["id"],
                series=custom.get("series", "Custom"),
                model=custom["model"],
                cruise_speed_kmh=custom["cruise_speed_kmh"],
                max_speed_kmh=custom.get("max_speed_kmh", custom["cruise_speed_kmh"]),
                max_flight_time_min=custom["max_flight_time_min"],
            )
        feasibility_result = check_feasibility(drone, route_distance_m)

    elif req.cruise_speed_kmh is not None and req.safe_flight_time_min is not None:
        # Path 2: inline constraints — no catalog lookup required.
        if req.cruise_speed_kmh <= 0 or req.safe_flight_time_min <= 0:
            raise HTTPException(
                status_code=400,
                detail="cruise_speed_kmh and safe_flight_time_min must be positive.",
            )
        cruise_ms = req.cruise_speed_kmh / 3.6
        estimated_time_min = (route_distance_m / cruise_ms) / 60.0
        safe_time = req.safe_flight_time_min
        feasible = estimated_time_min <= safe_time
        battery_pct = (
            min(estimated_time_min / safe_time * 100.0, 200.0) if safe_time > 0 else 200.0
        )
        feasibility_result = {
            "feasible":             feasible,
            "estimated_time_min":   round(estimated_time_min, 1),
            "safe_flight_time_min": round(safe_time, 1),
            "distance_km":          round(route_distance_m / 1000.0, 2),
            "battery_usage_pct":    round(battery_pct, 1),
            "reason": (
                ""
                if feasible
                else (
                    f"Mission needs ~{estimated_time_min:.1f} min but safe limit is "
                    f"{safe_time:.1f} min"
                )
            ),
        }

    response: dict = {
        "waypoints":        [{"lat": wp.lat, "lon": wp.lon, "alt": wp.alt} for wp in waypoints],
        "flight_time_sec":  round(flight_time_sec, 1),
        "total_waypoints":  len(waypoints),
        "route_distance_m": round(route_distance_m, 1),
    }
    # Only add the feasibility key when constraints were actually supplied so
    # that clients which send no drone constraints see an identical response
    # shape to the pre-constraint API.
    if feasibility_result is not None:
        response["feasibility"] = feasibility_result
    return response


class FlightParamsRecommendRequest(BaseModel):
    target_gsd_cm_px: float
    sensor_width_mm: float
    sensor_height_mm: float
    focal_length_mm: float
    image_width_px: int
    image_height_px: int
    drone_id: Optional[str] = None
    overlap_v: float = 0.7
    capture_interval_s: float = 2.0
    speed_safety_factor: float = 0.85


@router.post("/flight_params/recommend")
async def recommend_flight_params(req: FlightParamsRecommendRequest):
    """Task 5: Recommend altitude and flight speed for a target GSD."""
    from drone_oversight import AltitudeScaler

    positive_float_fields = {
        "target_gsd_cm_px": req.target_gsd_cm_px,
        "sensor_width_mm": req.sensor_width_mm,
        "sensor_height_mm": req.sensor_height_mm,
        "focal_length_mm": req.focal_length_mm,
        "capture_interval_s": req.capture_interval_s,
        "speed_safety_factor": req.speed_safety_factor,
    }
    for field_name, value in positive_float_fields.items():
        if value <= 0:
            raise HTTPException(status_code=400, detail=f"{field_name} must be positive.")

    if req.image_width_px <= 0 or req.image_height_px <= 0:
        raise HTTPException(status_code=400, detail="image_width_px and image_height_px must be positive.")
    if not (0.0 < req.overlap_v < 1.0):
        raise HTTPException(status_code=400, detail="overlap_v must be between 0 and 1.")
    if not (0.0 < req.speed_safety_factor <= 1.0):
        raise HTTPException(status_code=400, detail="speed_safety_factor must be between 0 and 1.")

    drone: Optional[DroneSpec] = None
    if req.drone_id:
        drone = _lookup_drone_spec(req.drone_id)

    target_gsd_m_px = req.target_gsd_cm_px / 100.0
    recommended_altitude_m = (
        target_gsd_m_px * req.focal_length_mm * req.image_width_px
    ) / req.sensor_width_mm

    horizontal_fov_deg = math.degrees(
        2.0 * math.atan(req.sensor_width_mm / (2.0 * req.focal_length_mm))
    )
    vertical_fov_deg = math.degrees(
        2.0 * math.atan(req.sensor_height_mm / (2.0 * req.focal_length_mm))
    )

    camera_config = CameraConfig(
        fov_h=horizontal_fov_deg,
        fov_v=vertical_fov_deg,
        gimbal_mode=GimbalMode.NADIR,
    )
    scaler = AltitudeScaler(
        reference_altitude_m=recommended_altitude_m,
        reference_gsd_m=target_gsd_m_px,
        camera_config=camera_config,
    )
    computed_gsd_m_px = scaler.compute_gsd(recommended_altitude_m, req.image_width_px)

    ground_height_m = (
        2.0 * recommended_altitude_m * math.tan(math.radians(vertical_fov_deg / 2.0))
    )
    track_advance_per_shot_m = ground_height_m * (1.0 - req.overlap_v)
    overlap_limited_speed_m_s = (
        track_advance_per_shot_m / req.capture_interval_s
    ) * req.speed_safety_factor

    explanation_parts = [
        f"Target GSD {req.target_gsd_cm_px:.2f} cm/px yields altitude {recommended_altitude_m:.1f} m",
        f"Vertical footprint {ground_height_m:.1f} m with {req.overlap_v * 100:.0f}% overlap and {req.capture_interval_s:.1f}s capture interval limits speed to {overlap_limited_speed_m_s:.1f} m/s",
    ]

    constraints = {
        "target_gsd_cm_px": round(req.target_gsd_cm_px, 3),
        "vertical_overlap_pct": round(req.overlap_v * 100.0, 1),
        "capture_interval_s": round(req.capture_interval_s, 2),
        "speed_safety_factor": round(req.speed_safety_factor, 2),
        "overlap_limited_speed_m_s": round(overlap_limited_speed_m_s, 2),
    }

    recommended_speed_m_s = overlap_limited_speed_m_s
    if drone is not None:
        cruise_speed_m_s = drone.cruise_speed_kmh / 3.6
        max_speed_m_s = drone.max_speed_kmh / 3.6
        recommended_speed_m_s = min(overlap_limited_speed_m_s, cruise_speed_m_s, max_speed_m_s)
        constraints["drone_model"] = drone.model
        constraints["drone_cruise_speed_m_s"] = round(cruise_speed_m_s, 2)
        constraints["drone_max_speed_m_s"] = round(max_speed_m_s, 2)
        explanation_parts.append(
            f"Drone limits cap the recommendation at {recommended_speed_m_s:.1f} m/s"
        )

    return {
        "target_gsd_cm_px": round(req.target_gsd_cm_px, 3),
        "recommended_altitude_m": round(recommended_altitude_m, 1),
        "recommended_speed_m_s": round(recommended_speed_m_s, 2),
        "recommended_speed_kmh": round(recommended_speed_m_s * 3.6, 1),
        "computed_gsd_cm_px": round(computed_gsd_m_px * 100.0, 3),
        "camera": {
            "sensor_width_mm": req.sensor_width_mm,
            "sensor_height_mm": req.sensor_height_mm,
            "focal_length_mm": req.focal_length_mm,
            "image_width_px": req.image_width_px,
            "image_height_px": req.image_height_px,
            "horizontal_fov_deg": round(horizontal_fov_deg, 3),
            "vertical_fov_deg": round(vertical_fov_deg, 3),
        },
        "constraints": constraints,
        "explanation": ". ".join(explanation_parts) + ".",
        "drone": (drone.to_dict() if drone is not None else None),
    }
