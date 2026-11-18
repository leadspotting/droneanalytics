"""
test_project_route.py  —  Integration tests for GET /mission/{id}/project
                          and GET /image_mission/{id}/project
==========================================================================
Run with:
    python -m pytest test_project_route.py -v
"""

import unittest
from fastapi.testclient import TestClient
import api as api_module


# Minimal two-row CSV that produces a MOSAIC_TEST mission with 2 valid footprints.
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


class TestProjectRoute(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved = api_module._missions.copy()

    def tearDown(self):
        api_module._missions.clear()
        api_module._missions.update(self._saved)
    def _load_two_row(self):
        r = self.client.post(
            "/upload",
            files={"file": ("proj.csv", _TWO_ROW_CSV.encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 200)
        # Cache canvas dims and bounding box from /mosaic for use in tests.
        m = self.client.get("/mission/MOSAIC_TEST/mosaic").json()
        self._W = m["canvas_width_px"]
        self._H = m["canvas_height_px"]
        self._bb = m["bounding_box"]

    # ------------------------------------------------------------------
    # 404 / 422
    # ------------------------------------------------------------------

    def test_unknown_mission_returns_404(self):
        r = self.client.get("/mission/DOES_NOT_EXIST/project?x=0&y=0")
        self.assertEqual(r.status_code, 404)

    def test_empty_mission_returns_422(self):
        api_module._missions["EMPTY_M"] = []
        r = self.client.get("/mission/EMPTY_M/project?x=0&y=0")
        self.assertEqual(r.status_code, 422)

    # ------------------------------------------------------------------
    # 400 out-of-bounds pixels
    # ------------------------------------------------------------------

    def test_negative_x_returns_400(self):
        self._load_two_row()
        r = self.client.get("/mission/MOSAIC_TEST/project?x=-1&y=0")
        self.assertEqual(r.status_code, 400)

    def test_negative_y_returns_400(self):
        self._load_two_row()
        r = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=-1")
        self.assertEqual(r.status_code, 400)

    def test_x_at_canvas_width_returns_400(self):
        self._load_two_row()
        r = self.client.get(f"/mission/MOSAIC_TEST/project?x={self._W}&y=0")
        self.assertEqual(r.status_code, 400)

    def test_y_at_canvas_height_returns_400(self):
        self._load_two_row()
        r = self.client.get(f"/mission/MOSAIC_TEST/project?x=0&y={self._H}")
        self.assertEqual(r.status_code, 400)

    # ------------------------------------------------------------------
    # 200 response structure
    # ------------------------------------------------------------------

    def test_returns_200_for_valid_pixel(self):
        self._load_two_row()
        r = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0")
        self.assertEqual(r.status_code, 200)

    def test_response_contains_required_keys(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        for key in (
            "mission_id", "x", "y", "lat", "lon",
            "canvas_width_px", "canvas_height_px", "gsd_m",
        ):
            self.assertIn(key, data, msg=f"Missing key: {key}")

    def test_mission_id_echoed(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        self.assertEqual(data["mission_id"], "MOSAIC_TEST")

    def test_x_y_echoed(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=10&y=5").json()
        self.assertEqual(data["x"], 10)
        self.assertEqual(data["y"], 5)

    def test_gsd_m_is_positive(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        self.assertGreater(data["gsd_m"], 0.0)

    def test_canvas_dims_are_positive_integers(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        self.assertIsInstance(data["canvas_width_px"], int)
        self.assertIsInstance(data["canvas_height_px"], int)
        self.assertGreater(data["canvas_width_px"], 0)
        self.assertGreater(data["canvas_height_px"], 0)

    def test_canvas_dims_match_mosaic_route(self):
        """/project canvas dims must agree with /mosaic at the same image_width_px."""
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        self.assertEqual(data["canvas_width_px"], self._W)
        self.assertEqual(data["canvas_height_px"], self._H)

    # ------------------------------------------------------------------
    # Geometric correctness
    # ------------------------------------------------------------------

    def test_top_left_pixel_maps_to_nw_corner(self):
        """Pixel (0, 0) must map to the NW (highest lat, lowest lon) corner."""
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        self.assertAlmostEqual(data["lat"], self._bb["TL"]["lat"], places=5)
        self.assertAlmostEqual(data["lon"], self._bb["TL"]["lon"], places=5)

    def test_bottom_right_pixel_maps_to_se_corner(self):
        """Pixel (W-1, H-1) must map to the SE (lowest lat, highest lon) corner."""
        self._load_two_row()
        data = self.client.get(
            f"/mission/MOSAIC_TEST/project?x={self._W - 1}&y={self._H - 1}"
        ).json()
        self.assertAlmostEqual(data["lat"], self._bb["BR"]["lat"], places=5)
        self.assertAlmostEqual(data["lon"], self._bb["BR"]["lon"], places=5)

    def test_reverse_project_nw_corner_returns_top_left_pixel(self):
        data = self.client.get(
            f"/image_mission/IMG_TEST/project/reverse?lat={self._bb['TL']['lat']}&lon={self._bb['TL']['lon']}"
        ).json()
        self.assertEqual(data["x"], 0)
        self.assertEqual(data["y"], 0)

    def test_reverse_project_outside_bbox_returns_400(self):
        r = self.client.get("/image_mission/IMG_TEST/project/reverse?lat=90&lon=0")
        self.assertEqual(r.status_code, 400)

    def test_project_round_trip_via_reverse_route(self):
        x_in, y_in = self._W // 2, self._H // 2
        gps = self.client.get(
            f"/image_mission/IMG_TEST/project?x={x_in}&y={y_in}"
        ).json()
        px = self.client.get(
            f"/image_mission/IMG_TEST/project/reverse?lat={gps['lat']}&lon={gps['lon']}"
        ).json()
        self.assertEqual(px["x"], x_in)
        self.assertEqual(px["y"], y_in)

    def test_lat_in_valid_gps_range(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        self.assertGreaterEqual(data["lat"], -90.0)
        self.assertLessEqual(data["lat"], 90.0)

    def test_lon_in_valid_gps_range(self):
        self._load_two_row()
        data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        self.assertGreaterEqual(data["lon"], -180.0)
        self.assertLessEqual(data["lon"], 180.0)

    def test_custom_image_width_changes_canvas_size(self):
        """Larger image_width_px → finer GSD → larger canvas."""
        self._load_two_row()
        d4000 = self.client.get(
            "/mission/MOSAIC_TEST/project?x=0&y=0&image_width_px=4000"
        ).json()
        d1000 = self.client.get(
            "/mission/MOSAIC_TEST/project?x=0&y=0&image_width_px=1000"
        ).json()
        self.assertGreater(d4000["canvas_width_px"], d1000["canvas_width_px"])

    def test_reverse_project_nw_corner_returns_top_left_pixel(self):
        self._load_two_row()
        data = self.client.get(
            f"/mission/MOSAIC_TEST/project/reverse?lat={self._bb['TL']['lat']}&lon={self._bb['TL']['lon']}"
        ).json()
        self.assertEqual(data["x"], 0)
        self.assertEqual(data["y"], 0)

    def test_reverse_project_outside_bbox_returns_400(self):
        self._load_two_row()
        r = self.client.get("/mission/MOSAIC_TEST/project/reverse?lat=90&lon=0")
        self.assertEqual(r.status_code, 400)

    def test_project_round_trip_via_reverse_route(self):
        self._load_two_row()
        x_in, y_in = self._W // 2, self._H // 2
        gps = self.client.get(
            f"/mission/MOSAIC_TEST/project?x={x_in}&y={y_in}"
        ).json()
        px = self.client.get(
            f"/mission/MOSAIC_TEST/project/reverse?lat={gps['lat']}&lon={gps['lon']}"
        ).json()
        self.assertEqual(px["x"], x_in)
        self.assertEqual(px["y"], y_in)


# ---------------------------------------------------------------------------
# Degenerate canvas guard
# ---------------------------------------------------------------------------

# Single-row CSV: with image_width_px=1 gsd_m ≈ ground_width_m (≈200m),
# which exceeds the single footprint span (≈115m) → canvas < 2px → 422.
_ONE_ROW_CSV = (
    "Drone_ID,Mission_ID,Operator_ID,Timestamp,Latitude,Longitude,Altitude_m,"
    "Speed_m_s,Battery_pct,Signal_pct,Mission_Type,Takeoff_Location,"
    "Landing_Location,Crash_Event,GPS_Tampering,Firmware_Version,"
    "Memory_Integrity,Data_Leak_Detected,Weather_Condition,Visibility_km,"
    "Wind_Speed_m_s,Temperature_C,Login_Attempts,Unauthorized_Access,"
    "Data_Encryption_Status,Network_Jamming\n"
    "DRN_0001,DEGEN_TELEMETRY,OP_001,2024-01-01 00:00:00,34.05,118.25,100,10,90,95,"
    "Mapping,LAX,LAX,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
)


class TestDegenerateCanvas(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved = api_module._missions.copy()

    def tearDown(self):
        api_module._missions.clear()
        api_module._missions.update(self._saved)
    def test_degenerate_canvas_returns_422(self):
        """image_width_px=1 → gsd_m≈200m; single-footprint lat span≈115m → H_canvas=1 → 422."""
        r = self.client.post(
            "/upload",
            files={"file": ("degen.csv", _ONE_ROW_CSV.encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 200)
        r = self.client.get("/mission/DEGEN_TELEMETRY/project?x=0&y=0&image_width_px=1")
        self.assertEqual(r.status_code, 422)


# ---------------------------------------------------------------------------
# Image mission project route
# ---------------------------------------------------------------------------

# Synthetic image rows (no real files needed — footprint math only uses GPS/FOV/agl_m).
_IMG_ROW_1 = {
    "path": "/fake/img_001.JPG",
    "lat": 36.340, "lon": -118.760,
    "agl_m": 110.0, "alt_ellipsoid_m": 1465.0,
    "timestamp": "2024-04-30T22:22:21Z",
    "heading": 0.0,
    "width_px": 4928, "height_px": 3264,
    "fov_h": 65.8, "fov_v": 46.4,
}
_IMG_ROW_2 = {
    "path": "/fake/img_002.JPG",
    "lat": 36.350, "lon": -118.750,
    "agl_m": 110.0, "alt_ellipsoid_m": 1465.0,
    "timestamp": "2024-04-30T22:22:22Z",
    "heading": 45.0,
    "width_px": 4928, "height_px": 3264,
    "fov_h": 65.8, "fov_v": 46.4,
}


class TestImageProjectRoute(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved = api_module._image_missions.copy()
        api_module._image_missions["IMG_TEST"] = [_IMG_ROW_1, _IMG_ROW_2]
        m = self.client.get("/image_mission/IMG_TEST/mosaic").json()
        self._W = m["canvas_width_px"]
        self._H = m["canvas_height_px"]
        self._bb = m["bounding_box"]

    def tearDown(self):
        api_module._image_missions.clear()
        api_module._image_missions.update(self._saved)
    # ------------------------------------------------------------------
    # 404 / 422
    # ------------------------------------------------------------------

    def test_unknown_mission_returns_404(self):
        r = self.client.get("/image_mission/DOES_NOT_EXIST/project?x=0&y=0")
        self.assertEqual(r.status_code, 404)

    def test_empty_mission_returns_422(self):
        api_module._image_missions["EMPTY_IMG"] = []
        r = self.client.get("/image_mission/EMPTY_IMG/project?x=0&y=0")
        self.assertEqual(r.status_code, 422)

    def test_degenerate_canvas_returns_422(self):
        """Single image row + image_width_px=1 → canvas < 2px → 422."""
        api_module._image_missions["DEGEN_IMG"] = [_IMG_ROW_1]
        r = self.client.get("/image_mission/DEGEN_IMG/project?x=0&y=0&image_width_px=1")
        self.assertEqual(r.status_code, 422)

    # ------------------------------------------------------------------
    # 400 out-of-bounds pixels
    # ------------------------------------------------------------------

    def test_negative_x_returns_400(self):
        r = self.client.get("/image_mission/IMG_TEST/project?x=-1&y=0")
        self.assertEqual(r.status_code, 400)

    def test_negative_y_returns_400(self):
        r = self.client.get("/image_mission/IMG_TEST/project?x=0&y=-1")
        self.assertEqual(r.status_code, 400)

    def test_x_at_canvas_width_returns_400(self):
        r = self.client.get(f"/image_mission/IMG_TEST/project?x={self._W}&y=0")
        self.assertEqual(r.status_code, 400)

    def test_y_at_canvas_height_returns_400(self):
        r = self.client.get(f"/image_mission/IMG_TEST/project?x=0&y={self._H}")
        self.assertEqual(r.status_code, 400)

    # ------------------------------------------------------------------
    # 200 response structure
    # ------------------------------------------------------------------

    def test_returns_200_for_valid_pixel(self):
        r = self.client.get("/image_mission/IMG_TEST/project?x=0&y=0")
        self.assertEqual(r.status_code, 200)

    def test_response_contains_required_keys(self):
        data = self.client.get("/image_mission/IMG_TEST/project?x=0&y=0").json()
        for key in (
            "mission_id", "x", "y", "lat", "lon",
            "canvas_width_px", "canvas_height_px", "gsd_m",
        ):
            self.assertIn(key, data, msg=f"Missing key: {key}")

    def test_mission_id_echoed(self):
        data = self.client.get("/image_mission/IMG_TEST/project?x=0&y=0").json()
        self.assertEqual(data["mission_id"], "IMG_TEST")

    def test_x_y_echoed(self):
        data = self.client.get("/image_mission/IMG_TEST/project?x=10&y=5").json()
        self.assertEqual(data["x"], 10)
        self.assertEqual(data["y"], 5)

    def test_canvas_dims_match_mosaic_route(self):
        data = self.client.get("/image_mission/IMG_TEST/project?x=0&y=0").json()
        self.assertEqual(data["canvas_width_px"], self._W)
        self.assertEqual(data["canvas_height_px"], self._H)

    # ------------------------------------------------------------------
    # Geometric correctness
    # ------------------------------------------------------------------

    def test_top_left_pixel_maps_to_nw_corner(self):
        """Pixel (0, 0) must map to the NW (highest lat, lowest lon) corner."""
        data = self.client.get("/image_mission/IMG_TEST/project?x=0&y=0").json()
        self.assertAlmostEqual(data["lat"], self._bb["TL"]["lat"], places=5)
        self.assertAlmostEqual(data["lon"], self._bb["TL"]["lon"], places=5)

    def test_bottom_right_pixel_maps_to_se_corner(self):
        """Pixel (W-1, H-1) must map to the SE (lowest lat, highest lon) corner."""
        data = self.client.get(
            f"/image_mission/IMG_TEST/project?x={self._W - 1}&y={self._H - 1}"
        ).json()
        self.assertAlmostEqual(data["lat"], self._bb["BR"]["lat"], places=5)
        self.assertAlmostEqual(data["lon"], self._bb["BR"]["lon"], places=5)


# ---------------------------------------------------------------------------
# image_width_px validation (both routes)
# ---------------------------------------------------------------------------

class TestImageWidthValidation(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved_m = api_module._missions.copy()
        self._saved_im = api_module._image_missions.copy()
        # Load telemetry mission
        r = self.client.post(
            "/upload",
            files={"file": ("proj.csv", _TWO_ROW_CSV.encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 200)
        # Load image mission
        api_module._image_missions["IMG_VALID"] = [_IMG_ROW_1, _IMG_ROW_2]

    def tearDown(self):
        api_module._missions.clear()
        api_module._missions.update(self._saved_m)
        api_module._image_missions.clear()
        api_module._image_missions.update(self._saved_im)
    # -- /mission/{id}/project --

    def test_mission_project_zero_image_width_returns_400(self):
        r = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0&image_width_px=0")
        self.assertEqual(r.status_code, 400)

    def test_mission_project_negative_image_width_returns_400(self):
        r = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0&image_width_px=-1")
        self.assertEqual(r.status_code, 400)

    # -- /image_mission/{id}/project --

    def test_image_mission_project_zero_image_width_returns_400(self):
        r = self.client.get("/image_mission/IMG_VALID/project?x=0&y=0&image_width_px=0")
        self.assertEqual(r.status_code, 400)

    def test_image_mission_project_negative_image_width_returns_400(self):
        r = self.client.get("/image_mission/IMG_VALID/project?x=0&y=0&image_width_px=-1")
        self.assertEqual(r.status_code, 400)


# ---------------------------------------------------------------------------
# Response key parity between /mission/{id}/project and /image_mission/{id}/project
# ---------------------------------------------------------------------------

_REQUIRED_PROJECT_KEYS = (
    "mission_id", "x", "y", "lat", "lon",
    "canvas_width_px", "canvas_height_px", "gsd_m",
)


class TestProjectResponseParity(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved_m = api_module._missions.copy()
        self._saved_im = api_module._image_missions.copy()
        r = self.client.post(
            "/upload",
            files={"file": ("proj.csv", _TWO_ROW_CSV.encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 200)
        api_module._image_missions["IMG_PARITY"] = [_IMG_ROW_1, _IMG_ROW_2]

    def tearDown(self):
        api_module._missions.clear()
        api_module._missions.update(self._saved_m)
        api_module._image_missions.clear()
        api_module._image_missions.update(self._saved_im)
    def test_both_routes_return_same_top_level_keys(self):
        m_data = self.client.get("/mission/MOSAIC_TEST/project?x=0&y=0").json()
        i_data = self.client.get("/image_mission/IMG_PARITY/project?x=0&y=0").json()
        for key in _REQUIRED_PROJECT_KEYS:
            self.assertIn(key, m_data, msg=f"/mission route missing key: {key}")
            self.assertIn(key, i_data, msg=f"/image_mission route missing key: {key}")
        self.assertEqual(set(m_data.keys()), set(i_data.keys()))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
