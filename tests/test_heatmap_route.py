import unittest

from fastapi.testclient import TestClient

import api as api_module


_MULTI_DRONE_CSV = (
    "Drone_ID,Mission_ID,Operator_ID,Timestamp,Latitude,Longitude,Altitude_m,"
    "Speed_m_s,Battery_pct,Signal_pct,Mission_Type,Takeoff_Location,"
    "Landing_Location,Crash_Event,GPS_Tampering,Firmware_Version,"
    "Memory_Integrity,Data_Leak_Detected,Weather_Condition,Visibility_km,"
    "Wind_Speed_m_s,Temperature_C,Login_Attempts,Unauthorized_Access,"
    "Data_Encryption_Status,Network_Jamming\n"
    "DRN_A,HEATMAP_MULTI,OP_001,2024-01-01 00:00:00,34.0500,118.2500,100,10,90,95,"
    "Mapping,LAX,LAX,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
    "DRN_A,HEATMAP_MULTI,OP_001,2024-01-01 00:00:01,34.0504,118.2504,100,10,90,95,"
    "Mapping,LAX,LAX,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
    "DRN_B,HEATMAP_MULTI,OP_001,2024-01-01 00:00:02,34.0650,118.2650,100,10,90,95,"
    "Mapping,LAX,LAX,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
    "DRN_B,HEATMAP_MULTI,OP_001,2024-01-01 00:00:03,34.0654,118.2654,100,10,90,95,"
    "Mapping,LAX,LAX,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
)


class TestHeatmapRoute(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved = api_module._missions.copy()

    def tearDown(self):
        api_module._missions.clear()
        api_module._missions.update(self._saved)
    def _load_multi_drone(self):
        response = self.client.post(
            "/upload",
            files={"file": ("heatmap_multi.csv", _MULTI_DRONE_CSV.encode(), "text/csv")},
        )
        self.assertEqual(response.status_code, 200)

    def test_heatmap_without_filter_preserves_existing_shape(self):
        self._load_multi_drone()
        response = self.client.get("/mission/HEATMAP_MULTI/heatmap?resolution_m=150")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mission_id"], "HEATMAP_MULTI")
        self.assertIsNone(data["drone_filter_applied"])
        self.assertIn("coverage_pct", data)
        self.assertIn("max_overlap", data)
        self.assertIn("cells", data)

    def test_heatmap_with_drone_filter_changes_results(self):
        self._load_multi_drone()
        all_data = self.client.get("/mission/HEATMAP_MULTI/heatmap?resolution_m=150").json()
        drn_a = self.client.get(
            "/mission/HEATMAP_MULTI/heatmap?resolution_m=150&drone_id=DRN_A"
        ).json()
        drn_b = self.client.get(
            "/mission/HEATMAP_MULTI/heatmap?resolution_m=150&drone_id=DRN_B"
        ).json()

        self.assertEqual(drn_a["drone_filter_applied"], "DRN_A")
        self.assertEqual(drn_b["drone_filter_applied"], "DRN_B")
        self.assertNotEqual(all_data["cells"], drn_a["cells"])
        self.assertNotEqual(all_data["cells"], drn_b["cells"])
        self.assertNotEqual(drn_a["cells"], drn_b["cells"])

    def test_unknown_drone_filter_returns_empty_coverage(self):
        self._load_multi_drone()
        data = self.client.get(
            "/mission/HEATMAP_MULTI/heatmap?resolution_m=150&drone_id=DOES_NOT_EXIST"
        ).json()
        self.assertEqual(data["drone_filter_applied"], "DOES_NOT_EXIST")
        self.assertEqual(data["coverage_pct"], 0.0)
        self.assertEqual(data["max_overlap"], 0)
        self.assertEqual(data["cells"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
