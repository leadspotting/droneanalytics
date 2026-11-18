"""
test_plan_mission_constraints.py
=================================
Tests for the constraint-aware /plan_mission endpoint (Task 6).

Covers:
  - Backward compatibility: no drone constraints → feasibility is null
  - Catalog drone path (drone_id): feasible and infeasible cases
  - Inline constraint path (cruise_speed_kmh + safe_flight_time_min):
      feasible and infeasible cases
  - Unknown drone_id → 400 with helpful message
  - Response contract: all expected keys present when constraints applied

The geometry tests (strip spacing, serpentine order, edge coverage, etc.)
live in test_mission_planner.py and are not duplicated here.
"""

import unittest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixture — a small (~100 m × 100 m) polygon in Tel Aviv.
# At 100 m altitude, 90°×60° FOV, 70% overlap this takes only a few seconds
# to fly, so any sensible drone from the catalog will pass feasibility.
# ---------------------------------------------------------------------------
SMALL_POLYGON = [
    {"lat": 32.0800, "lon": 34.7800},
    {"lat": 32.0800, "lon": 34.7811},
    {"lat": 32.0809, "lon": 34.7811},
    {"lat": 32.0809, "lon": 34.7800},
]

# A large polygon (~60 km × 60 km) that will produce a route requiring many
# hours of flight time — guaranteed infeasible for any real drone.
HUGE_POLYGON = [          # ~6 km x 5.6 km: far beyond any drone battery, still a sane plan size
    {"lat": 32.00, "lon": 34.00},
    {"lat": 32.00, "lon": 34.06},
    {"lat": 32.06, "lon": 34.06},
    {"lat": 32.06, "lon": 34.00},
]

ABSURD_POLYGON = [        # ~60 km x 60 km: rejected instead of freezing the UI
    {"lat": 32.0, "lon": 34.0},
    {"lat": 32.0, "lon": 34.6},
    {"lat": 32.54, "lon": 34.6},
    {"lat": 32.54, "lon": 34.0},
]

BASE_PAYLOAD = {
    "altitude_m": 100.0,
    "overlap_h":  0.7,
    "overlap_v":  0.7,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def plan(polygon, extra=None):
    payload = {"area": polygon, **BASE_PAYLOAD, **(extra or {})}
    return client.post("/plan_mission", json=payload)


# ---------------------------------------------------------------------------
# 1. Backward compatibility — no constraints supplied
# ---------------------------------------------------------------------------

class TestNoDroneConstraints(unittest.TestCase):

    def test_response_200(self):
        r = plan(SMALL_POLYGON)
        self.assertEqual(r.status_code, 200)

    def test_feasibility_key_absent(self):
        """Omitting all constraint fields must omit the feasibility key entirely."""
        data = plan(SMALL_POLYGON).json()
        self.assertNotIn("feasibility", data)

    def test_legacy_fields_present(self):
        """flight_time_sec, total_waypoints, route_distance_m must still exist."""
        data = plan(SMALL_POLYGON).json()
        for key in ("waypoints", "flight_time_sec", "total_waypoints", "route_distance_m"):
            self.assertIn(key, data, f"Missing legacy key: {key}")

    def test_waypoints_non_empty(self):
        data = plan(SMALL_POLYGON).json()
        self.assertGreater(len(data["waypoints"]), 0)


# ---------------------------------------------------------------------------
# 2. Catalog drone path — drone_id
# ---------------------------------------------------------------------------

class TestCatalogDrone(unittest.TestCase):

    def test_feasible_small_aoi_mavic4pro(self):
        """Small AOI + DJI Mavic 4 Pro (40.8 min safe) must be feasible."""
        data = plan(SMALL_POLYGON, {"drone_id": "mavic-4-pro"}).json()
        f = data["feasibility"]
        self.assertIsNotNone(f)
        self.assertTrue(f["feasible"], f"Expected feasible; reason: {f.get('reason')}")
        self.assertEqual(f["reason"], "")
        self.assertGreater(f["safe_flight_time_min"], 0)
        self.assertGreaterEqual(f["battery_usage_pct"], 0)

    def test_infeasible_large_aoi_avata2(self):
        """Huge AOI + DJI Avata 2 (18.4 min safe) must be infeasible."""
        data = plan(HUGE_POLYGON, {"drone_id": "avata-2"}).json()
        f = data["feasibility"]
        self.assertIsNotNone(f)
        self.assertFalse(f["feasible"])
        self.assertNotEqual(f["reason"], "", "reason must be non-empty when infeasible")
        self.assertGreater(f["estimated_time_min"], f["safe_flight_time_min"])
        self.assertGreater(f["battery_usage_pct"], 100.0)

    def test_route_still_returned_when_infeasible(self):
        """The route must always be returned, even when infeasible."""
        data = plan(HUGE_POLYGON, {"drone_id": "avata-2"}).json()
        self.assertIn("waypoints", data)
        self.assertGreater(len(data["waypoints"]), 0)
        self.assertGreater(data["route_distance_m"], 0)

    def test_absurdly_large_area_is_rejected_quickly(self):
        r = plan(ABSURD_POLYGON, {"drone_id": "avata-2"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("too large", r.json()["detail"])

    def test_feasibility_keys_complete(self):
        """feasibility dict must contain all expected keys."""
        data = plan(SMALL_POLYGON, {"drone_id": "mini-4-pro"}).json()
        f = data["feasibility"]
        for key in (
            "feasible",
            "estimated_time_min",
            "safe_flight_time_min",
            "distance_km",
            "battery_usage_pct",
            "reason",
        ):
            self.assertIn(key, f, f"Missing feasibility key: {key}")

    def test_unknown_drone_id_returns_400(self):
        r = plan(SMALL_POLYGON, {"drone_id": "totally-fake-drone-xyz"})
        self.assertEqual(r.status_code, 400)
        detail = r.json()["detail"]
        self.assertIn("totally-fake-drone-xyz", detail)
        self.assertIn("/drones", detail)  # hint about where to look

    def test_distance_km_consistent_with_route_distance(self):
        """feasibility.distance_km must equal route_distance_m / 1000 (rounded)."""
        data = plan(SMALL_POLYGON, {"drone_id": "air-3"}).json()
        f = data["feasibility"]
        expected_km = round(data["route_distance_m"] / 1000.0, 2)
        self.assertAlmostEqual(f["distance_km"], expected_km, places=2)


# ---------------------------------------------------------------------------
# 3. Inline constraint path — cruise_speed_kmh + safe_flight_time_min
# ---------------------------------------------------------------------------

class TestInlineConstraints(unittest.TestCase):

    def test_feasible_with_huge_endurance(self):
        """Effectively infinite endurance must always report feasible."""
        data = plan(
            SMALL_POLYGON,
            {"cruise_speed_kmh": 50.0, "safe_flight_time_min": 9999.0},
        ).json()
        f = data["feasibility"]
        self.assertIsNotNone(f)
        self.assertTrue(f["feasible"])
        self.assertEqual(f["reason"], "")

    def test_infeasible_with_tiny_endurance(self):
        """Effectively zero endurance (0.01 min) must always report infeasible."""
        data = plan(
            SMALL_POLYGON,
            {"cruise_speed_kmh": 50.0, "safe_flight_time_min": 0.01},
        ).json()
        f = data["feasibility"]
        self.assertIsNotNone(f)
        self.assertFalse(f["feasible"])
        self.assertNotEqual(f["reason"], "", "reason must be non-empty when infeasible")

    def test_reason_includes_safe_time(self):
        # 0.1 min safe time renders as "0.1" via :.1f and is below the ~0.3 min
        # estimated time for the small polygon → infeasible, reason non-empty.
        data = plan(
            SMALL_POLYGON,
            {"cruise_speed_kmh": 50.0, "safe_flight_time_min": 0.1},
        ).json()
        f = data["feasibility"]
        self.assertFalse(f["feasible"])
        self.assertIn("0.1", f["reason"])

    def test_battery_pct_capped_at_200(self):
        """battery_usage_pct must be capped at 200 even for extreme infeasibility."""
        data = plan(
            HUGE_POLYGON,
            {"cruise_speed_kmh": 1.0, "safe_flight_time_min": 0.001},
        ).json()
        self.assertLessEqual(data["feasibility"]["battery_usage_pct"], 200.0)

    def test_inline_negative_speed_returns_400(self):
        r = plan(SMALL_POLYGON, {"cruise_speed_kmh": -10.0, "safe_flight_time_min": 30.0})
        self.assertEqual(r.status_code, 400)

    def test_inline_negative_endurance_returns_400(self):
        r = plan(SMALL_POLYGON, {"cruise_speed_kmh": 50.0, "safe_flight_time_min": -5.0})
        self.assertEqual(r.status_code, 400)

    def test_partial_inline_constraints_ignored(self):
        """Providing only one of the two inline fields must behave like no constraints."""
        # Only cruise_speed_kmh, no safe_flight_time_min → falls through to no-constraint path
        # feasibility key must be absent entirely (not null) — same as unconstrained callers.
        data = plan(SMALL_POLYGON, {"cruise_speed_kmh": 50.0}).json()
        self.assertNotIn("feasibility", data)

    def test_feasibility_key_present_when_constrained(self):
        """feasibility key must be present when any constraint is supplied."""
        data = plan(SMALL_POLYGON, {"cruise_speed_kmh": 50.0, "safe_flight_time_min": 999.0}).json()
        self.assertIn("feasibility", data)


# ---------------------------------------------------------------------------
# 4. drone_id takes precedence over inline constraints
# ---------------------------------------------------------------------------

class TestDroneIdPriority(unittest.TestCase):

    def test_drone_id_wins_over_inline(self):
        """When both drone_id and inline fields are given, drone_id wins."""
        # Use mavic-4-pro (feasible for small AOI) but give inline safe_time=0.001
        # (which would force infeasible).  drone_id path should win → feasible.
        data = plan(
            SMALL_POLYGON,
            {
                "drone_id": "mavic-4-pro",
                "cruise_speed_kmh": 50.0,
                "safe_flight_time_min": 0.001,
            },
        ).json()
        f = data["feasibility"]
        # drone_id path: mavic-4-pro safe time = 51 * 0.8 = 40.8 min → small AOI feasible
        self.assertTrue(f["feasible"])


if __name__ == "__main__":
    unittest.main()
