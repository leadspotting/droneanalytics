import unittest

from fastapi.testclient import TestClient

import api as api_module


client = TestClient(api_module.app)


class TestFlightParamsRecommendRoute(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "target_gsd_cm_px": 2.5,
            "sensor_width_mm": 13.2,
            "sensor_height_mm": 8.8,
            "focal_length_mm": 8.8,
            "image_width_px": 5472,
            "image_height_px": 3648,
            "overlap_v": 0.7,
            "capture_interval_s": 2.0,
            "drone_id": "mavic-4-pro",
        }
        payload.update(overrides)
        return payload

    def test_recommend_returns_expected_shape(self):
        response = client.post("/flight_params/recommend", json=self._payload())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in (
            "recommended_altitude_m",
            "recommended_speed_m_s",
            "recommended_speed_kmh",
            "computed_gsd_cm_px",
            "constraints",
            "explanation",
        ):
            self.assertIn(key, data)

    def test_smaller_target_gsd_reduces_altitude(self):
        coarse = client.post("/flight_params/recommend", json=self._payload(target_gsd_cm_px=4.0)).json()
        fine = client.post("/flight_params/recommend", json=self._payload(target_gsd_cm_px=2.0)).json()
        self.assertLess(fine["recommended_altitude_m"], coarse["recommended_altitude_m"])

    def test_drone_limits_cap_speed(self):
        limited = client.post("/flight_params/recommend", json=self._payload(drone_id="avata-2", target_gsd_cm_px=6.0)).json()
        unlimited = client.post("/flight_params/recommend", json=self._payload(drone_id=None, target_gsd_cm_px=6.0)).json()
        self.assertLessEqual(limited["recommended_speed_m_s"], unlimited["recommended_speed_m_s"])

    def test_unknown_drone_returns_404(self):
        response = client.post("/flight_params/recommend", json=self._payload(drone_id="ghost-drone"))
        self.assertEqual(response.status_code, 404)

    def test_invalid_numeric_values_return_400(self):
        response = client.post("/flight_params/recommend", json=self._payload(sensor_width_mm=0))
        self.assertEqual(response.status_code, 400)

    def test_overlap_changes_speed(self):
        lower_overlap = client.post("/flight_params/recommend", json=self._payload(overlap_v=0.6)).json()
        higher_overlap = client.post("/flight_params/recommend", json=self._payload(overlap_v=0.85)).json()
        self.assertGreater(lower_overlap["recommended_speed_m_s"], higher_overlap["recommended_speed_m_s"])


if __name__ == "__main__":
    unittest.main()
