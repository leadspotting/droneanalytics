"""
test_mosaic_route.py  —  Integration tests for GET /mission/{id}/mosaic
========================================================================
Run with:
    python -m pytest test_mosaic_route.py -v
"""

import unittest
from fastapi.testclient import TestClient
import api as api_module


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Minimal CSV that produces two valid footprints (LA_MISSION_TEST)
_TWO_ROW_CSV = (
    "Drone_ID,Mission_ID,Operator_ID,Timestamp,Latitude,Longitude,Altitude_m,"
    "Speed_m_s,Battery_pct,Signal_pct,Mission_Type,Takeoff_Location,"
    "Landing_Location,Crash_Event,GPS_Tampering,Firmware_Version,"
    "Memory_Integrity,Data_Leak_Detected,Weather_Condition,Visibility_km,"
    "Wind_Speed_m_s,Temperature_C,Login_Attempts,Unauthorized_Access,"
    "Data_Encryption_Status,Network_Jamming\n"
    "DRN_0001,MOSAIC_TEST,OP_001,2024-01-01 00:00:00,34.05,118.25,100,10,90,95,"
    "Mapping,LAX,LAX,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
    "DRN_0001,MOSAIC_TEST,OP_001,2024-01-01 00:00:01,34.06,118.26,100,10,90,95,"
    "Mapping,LAX,LAX,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
)

# Minimal CSV with a single row (for single-footprint tests)
_ONE_ROW_CSV = (
    "Drone_ID,Mission_ID,Operator_ID,Timestamp,Latitude,Longitude,Altitude_m,"
    "Speed_m_s,Battery_pct,Signal_pct,Mission_Type,Takeoff_Location,"
    "Landing_Location,Crash_Event,GPS_Tampering,Firmware_Version,"
    "Memory_Integrity,Data_Leak_Detected,Weather_Condition,Visibility_km,"
    "Wind_Speed_m_s,Temperature_C,Login_Attempts,Unauthorized_Access,"
    "Data_Encryption_Status,Network_Jamming\n"
    "DRN_0001,SINGLE_FP,OP_001,2024-01-01 00:00:00,32.08,34.78,50,10,90,95,"
    "Mapping,TLV,TLV,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
)


class TestMosaicRoute(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved = api_module._missions.copy()

    def tearDown(self):
        api_module._missions.clear()
        api_module._missions.update(self._saved)
    # ------------------------------------------------------------------
    # 404 / 422
    # ------------------------------------------------------------------

    def test_unknown_mission_returns_404(self):
        r = self.client.get("/mission/DOES_NOT_EXIST/mosaic")
        self.assertEqual(r.status_code, 404)

    def test_mission_with_no_rows_returns_422(self):
        api_module._missions["EMPTY_M"] = []
        r = self.client.get("/mission/EMPTY_M/mosaic")
        self.assertEqual(r.status_code, 422)

    # ------------------------------------------------------------------
    # Response structure
    # ------------------------------------------------------------------

    def _load_two_row(self):
        r = self.client.post(
            "/upload",
            files={"file": ("mosaic_test.csv", _TWO_ROW_CSV.encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 200)

    def test_returns_200_for_valid_mission(self):
        self._load_two_row()
        r = self.client.get("/mission/MOSAIC_TEST/mosaic")
        self.assertEqual(r.status_code, 200)

    def test_response_contains_required_keys(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        for key in (
            "mission_id", "status", "message", "footprint_count",
            "bounding_box", "gsd_m", "canvas_width_px", "canvas_height_px",
            "image_width_px",
        ):
            self.assertIn(key, data, msg=f"Missing key: {key}")

    def test_status_is_metadata_only(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        self.assertEqual(data["status"], "metadata_only")

    def test_mission_id_echoed_correctly(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        self.assertEqual(data["mission_id"], "MOSAIC_TEST")

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def test_bounding_box_has_four_corners(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        bb = data["bounding_box"]
        for corner in ("TL", "TR", "BR", "BL"):
            self.assertIn(corner, bb)
            self.assertIn("lat", bb[corner])
            self.assertIn("lon", bb[corner])

    def test_bounding_box_lat_ordering(self):
        """TL and TR must have larger lat than BR and BL."""
        self._load_two_row()
        bb = self.client.get("/mission/MOSAIC_TEST/mosaic").json()["bounding_box"]
        self.assertGreater(bb["TL"]["lat"], bb["BR"]["lat"])
        self.assertGreater(bb["TR"]["lat"], bb["BL"]["lat"])

    def test_bounding_box_lon_ordering(self):
        """TR and BR must have larger lon than TL and BL."""
        self._load_two_row()
        bb = self.client.get("/mission/MOSAIC_TEST/mosaic").json()["bounding_box"]
        self.assertGreater(bb["TR"]["lon"], bb["TL"]["lon"])
        self.assertGreater(bb["BR"]["lon"], bb["BL"]["lon"])

    def test_bounding_box_covers_both_telemetry_points(self):
        """Canvas must span at least from row-1 lat/lon to row-2 lat/lon."""
        self._load_two_row()
        bb = self.client.get("/mission/MOSAIC_TEST/mosaic").json()["bounding_box"]
        self.assertGreaterEqual(bb["TL"]["lat"], 34.06)   # northernmost row
        self.assertLessEqual(bb["BR"]["lat"], 34.05)      # southernmost row
        self.assertGreaterEqual(bb["TR"]["lon"], 118.26)  # easternmost row
        self.assertLessEqual(bb["TL"]["lon"], 118.25)     # westernmost row

    # ------------------------------------------------------------------
    # GSD and canvas dimensions
    # ------------------------------------------------------------------

    def test_gsd_is_positive(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        self.assertGreater(data["gsd_m"], 0.0)

    def test_canvas_dimensions_are_positive_integers(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        self.assertIsInstance(data["canvas_width_px"], int)
        self.assertIsInstance(data["canvas_height_px"], int)
        self.assertGreater(data["canvas_width_px"], 0)
        self.assertGreater(data["canvas_height_px"], 0)

    def test_default_image_width_is_4000(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        self.assertEqual(data["image_width_px"], 4000)

    def test_custom_image_width_accepted(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic?image_width_px=1000").json()
        self.assertEqual(data["image_width_px"], 1000)

    def test_smaller_image_width_gives_larger_gsd(self):
        """Fewer pixels across same footprint → coarser GSD."""
        self._load_two_row()
        gsd_4000 = self.client.get("/mission/MOSAIC_TEST/mosaic?image_width_px=4000").json()["gsd_m"]
        gsd_1000 = self.client.get("/mission/MOSAIC_TEST/mosaic?image_width_px=1000").json()["gsd_m"]
        self.assertGreater(gsd_1000, gsd_4000)

    def test_footprint_count_matches_row_count(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        self.assertEqual(data["footprint_count"], 2)

    # ------------------------------------------------------------------
    # Single-footprint mission
    # ------------------------------------------------------------------

    def test_single_footprint_mission_returns_200(self):
        r = self.client.post(
            "/upload",
            files={"file": ("single.csv", _ONE_ROW_CSV.encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 200)
        data = self.client.get("/mission/SINGLE_FP/mosaic").json()
        self.assertEqual(data["status"], "metadata_only")
        self.assertEqual(data["footprint_count"], 1)
        self.assertGreater(data["gsd_m"], 0)


class TestTelemetryHeadingUsage(unittest.TestCase):

    def test_row_to_footprint_uses_heading_field_when_present(self):
        row_north = {
            "Latitude": "32.08",
            "Longitude": "34.78",
            "Altitude_m": "100",
            "Heading": "0",
        }
        row_east = {
            "Latitude": "32.08",
            "Longitude": "34.78",
            "Altitude_m": "100",
            "Heading": "90",
        }

        fp_north = api_module._row_to_footprint(row_north)
        fp_east = api_module._row_to_footprint(row_east)

        self.assertIsNotNone(fp_north)
        self.assertIsNotNone(fp_east)
        self.assertNotEqual(fp_north["corners"], fp_east["corners"])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
