import unittest

from drone_catalog import BUILTIN_DRONES, DroneSpec, check_feasibility


class TestDroneCatalog(unittest.TestCase):

    # ------------------------------------------------------------------
    # Safe flight time formula
    # ------------------------------------------------------------------

    def test_safe_times_all_drones(self):
        """Every built-in drone must have safe_flight_time_min == max * 0.8."""
        for drone in BUILTIN_DRONES:
            expected = drone.max_flight_time_min * 0.8
            self.assertAlmostEqual(
                drone.safe_flight_time_min, expected, places=6,
                msg=f"{drone.model}: safe={drone.safe_flight_time_min} != {expected}",
            )

    def test_mini5pro_values(self):
        """DJI Mini 5 Pro: 52 min max → 41.6 min safe."""
        drone = next(d for d in BUILTIN_DRONES if d.id == "mini-5-pro")
        self.assertEqual(drone.max_flight_time_min, 52.0)
        self.assertAlmostEqual(drone.safe_flight_time_min, 41.6, places=5)

    def test_mini4pro_values(self):
        """DJI Mini 4 Pro: 45 min max → 36.0 min safe."""
        drone = next(d for d in BUILTIN_DRONES if d.id == "mini-4-pro")
        self.assertEqual(drone.max_flight_time_min, 45.0)
        self.assertAlmostEqual(drone.safe_flight_time_min, 36.0, places=5)

    def test_matrice350_values(self):
        """DJI Matrice 350 RTK: 55 min max → 44.0 min safe."""
        drone = next(d for d in BUILTIN_DRONES if d.id == "matrice-350-rtk")
        self.assertEqual(drone.max_flight_time_min, 55.0)
        self.assertAlmostEqual(drone.safe_flight_time_min, 44.0, places=5)

    def test_ids_are_unique(self):
        """All built-in drone IDs must be unique."""
        ids = [d.id for d in BUILTIN_DRONES]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate IDs found in BUILTIN_DRONES")

    def test_all_16_drones_present(self):
        self.assertEqual(len(BUILTIN_DRONES), 16)

    def test_to_dict_includes_required_keys(self):
        drone = BUILTIN_DRONES[0]
        d = drone.to_dict()
        for key in ("id", "series", "model", "cruise_speed_kmh", "max_speed_kmh",
                    "max_flight_time_min", "safe_flight_time_min", "custom"):
            self.assertIn(key, d, f"Key '{key}' missing from to_dict()")
        self.assertFalse(d["custom"])

    # ------------------------------------------------------------------
    # check_feasibility
    # ------------------------------------------------------------------

    def _mavic3pro(self) -> DroneSpec:
        return next(d for d in BUILTIN_DRONES if d.id == "mavic-3-pro")

    def test_feasibility_zero_distance(self):
        """Zero route distance is always feasible with 0% battery."""
        result = check_feasibility(self._mavic3pro(), 0.0)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["battery_usage_pct"], 0.0)
        self.assertEqual(result["estimated_time_min"], 0.0)
        self.assertEqual(result["reason"], "")

    def test_feasibility_feasible_short_route(self):
        """A 1 km route is well within any drone's safe battery."""
        result = check_feasibility(self._mavic3pro(), 1_000.0)
        self.assertTrue(result["feasible"])
        self.assertLess(result["battery_usage_pct"], 100.0)
        self.assertEqual(result["reason"], "")

    def test_feasibility_infeasible_huge_route(self):
        """A 500 km route is infeasible for every built-in drone."""
        for drone in BUILTIN_DRONES:
            result = check_feasibility(drone, 500_000.0)
            self.assertFalse(result["feasible"],
                             f"{drone.model} should be infeasible for 500 km")
            self.assertNotEqual(result["reason"], "",
                                f"{drone.model} infeasible result must include a reason")

    def test_feasibility_battery_pct_at_safe_limit(self):
        """
        Route exactly at the safe-time limit should give battery_usage_pct ~100%.
        Compute the route distance that takes exactly safe_flight_time_min at cruise speed.
        """
        drone = self._mavic3pro()
        # distance = cruise_speed_ms * safe_time_sec
        cruise_ms = drone.cruise_speed_kmh / 3.6
        exact_distance_m = cruise_ms * drone.safe_flight_time_min * 60.0
        result = check_feasibility(drone, exact_distance_m)
        # At the exact limit the mission is still feasible (≤ not <)
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["battery_usage_pct"], 100.0, places=3)

    def test_feasibility_battery_pct_capped_at_200(self):
        """battery_usage_pct must not exceed 200 even for absurd routes."""
        result = check_feasibility(self._mavic3pro(), 10_000_000.0)
        self.assertLessEqual(result["battery_usage_pct"], 200.0)

    def test_feasibility_distance_km_field(self):
        """distance_km must equal route_distance_m / 1000."""
        result = check_feasibility(self._mavic3pro(), 12_345.0)
        self.assertAlmostEqual(result["distance_km"], 12.345, places=2)

    def test_feasibility_reason_contains_model_name(self):
        """Infeasibility reason must mention the drone model."""
        drone = self._mavic3pro()
        result = check_feasibility(drone, 500_000.0)
        self.assertIn(drone.model, result["reason"])


if __name__ == "__main__":
    unittest.main()
