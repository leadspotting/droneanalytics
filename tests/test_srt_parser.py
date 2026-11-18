"""
test_srt_parser.py  —  Unit tests for Phase 3: DJI SRT parser
==============================================================
Run with:
    python -m pytest test_srt_parser.py -v
"""

import math
import unittest

from srt_parser import parse_srt, _bearing


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_block(
    frame: int = 1,
    timestamp: str = "2025-05-24 23:01:30.000",
    lat: float = 32.08000,
    lon: float = 34.78000,
    rel_alt: float = 50.0,
    abs_alt: float = 45.0,
    focal_len: float = 24.0,
) -> str:
    """Build a minimal but valid DJI SRT subtitle block."""
    return (
        f"{frame}\n"
        f"00:00:00,000 --> 00:00:00,016\n"
        f'<font size="28">FrameCnt: {frame}, DiffTime: 16ms\n'
        f"{timestamp}\n"
        f"[iso: 100] [shutter: 1/100.0] [fnum: 2.8] [ev: 0] [ct: 5000]"
        f" [color_md : default] [focal_len: {focal_len:.2f}]"
        f" [latitude: {lat:.5f}] [longitude: {lon:.5f}]"
        f" [rel_alt: {rel_alt:.3f} abs_alt: {abs_alt:.3f}] </font>"
    )


def _srt(*blocks: str) -> str:
    """Join SRT blocks with the required blank-line separator."""
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Test: basic parsing
# ---------------------------------------------------------------------------

class TestParseSRT(unittest.TestCase):

    def test_single_valid_block_yields_one_row(self):
        text = _srt(_make_block(lat=32.08, lon=34.78))
        result = parse_srt(text, "M1")
        self.assertIn("M1", result)
        self.assertEqual(len(result["M1"]), 1)

    def test_zero_gps_filtered_out(self):
        text = _srt(
            _make_block(lat=0.0, lon=0.0),
            _make_block(lat=0.0, lon=0.0),
        )
        result = parse_srt(text, "M1")
        self.assertEqual(len(result["M1"]), 0)

    def test_mixed_zero_and_valid_frames(self):
        text = _srt(
            _make_block(frame=1, lat=0.0, lon=0.0),
            _make_block(frame=2, lat=32.08, lon=34.78),
        )
        result = parse_srt(text, "M1")
        self.assertEqual(len(result["M1"]), 1)
        self.assertAlmostEqual(float(result["M1"][0]["Latitude"]), 32.08, places=4)

    def test_latitude_longitude_extracted_correctly(self):
        text = _srt(_make_block(lat=31.2345, lon=35.6789))
        row = parse_srt(text, "M1")["M1"][0]
        self.assertAlmostEqual(float(row["Latitude"]), 31.2345, places=4)
        self.assertAlmostEqual(float(row["Longitude"]), 35.6789, places=4)

    def test_altitude_uses_rel_alt_not_abs_alt(self):
        text = _srt(_make_block(rel_alt=75.5, abs_alt=200.0))
        row = parse_srt(text, "M1")["M1"][0]
        self.assertAlmostEqual(float(row["Altitude_m"]), 75.5, places=2)

    def test_timestamp_extracted(self):
        text = _srt(_make_block(timestamp="2025-06-01 08:30:00.000"))
        row = parse_srt(text, "M1")["M1"][0]
        self.assertEqual(row["Timestamp"], "2025-06-01 08:30:00.000")

    def test_mission_id_is_result_key(self):
        text = _srt(_make_block())
        result = parse_srt(text, "DJI_0044")
        self.assertIn("DJI_0044", result)

    def test_multiple_valid_blocks_all_returned(self):
        text = _srt(
            _make_block(frame=1, lat=32.08, lon=34.78),
            _make_block(frame=2, lat=32.09, lon=34.79),
            _make_block(frame=3, lat=32.10, lon=34.80),
        )
        result = parse_srt(text, "M1")
        self.assertEqual(len(result["M1"]), 3)

    def test_row_contains_required_fields(self):
        text = _srt(_make_block())
        row = parse_srt(text, "M1")["M1"][0]
        for field in (
            "Latitude", "Longitude", "Altitude_m", "Timestamp",
            "Mission_ID", "Drone_ID",
            "GPS_Tampering", "Data_Leak_Detected", "Memory_Integrity",
        ):
            self.assertIn(field, row, msg=f"Missing required field: {field}")

    def test_security_defaults(self):
        text = _srt(_make_block())
        row = parse_srt(text, "M1")["M1"][0]
        self.assertEqual(row["GPS_Tampering"], "0")
        self.assertEqual(row["Data_Leak_Detected"], "0")
        self.assertEqual(row["Memory_Integrity"], "OK")

    def test_drone_id_is_dji(self):
        text = _srt(_make_block())
        row = parse_srt(text, "M1")["M1"][0]
        self.assertEqual(row["Drone_ID"], "DJI")


# ---------------------------------------------------------------------------
# Test: heading derivation
# ---------------------------------------------------------------------------

class TestHeadingDerivation(unittest.TestCase):

    def test_first_frame_heading_is_zero(self):
        text = _srt(_make_block(lat=32.08, lon=34.78))
        row = parse_srt(text, "M1")["M1"][0]
        self.assertAlmostEqual(float(row["Heading"]), 0.0, places=6)

    def test_northward_movement_gives_heading_zero(self):
        """Second frame directly north → heading ≈ 0°."""
        text = _srt(
            _make_block(frame=1, lat=32.08, lon=34.78),
            _make_block(frame=2, lat=32.09, lon=34.78),
        )
        rows = parse_srt(text, "M1")["M1"]
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[1]["Heading"]), 0.0, delta=1.0)

    def test_eastward_movement_gives_heading_90(self):
        """Second frame directly east → heading ≈ 90°."""
        text = _srt(
            _make_block(frame=1, lat=32.08, lon=34.78),
            _make_block(frame=2, lat=32.08, lon=34.88),
        )
        rows = parse_srt(text, "M1")["M1"]
        self.assertAlmostEqual(float(rows[1]["Heading"]), 90.0, delta=1.0)

    def test_southward_movement_gives_heading_180(self):
        """Second frame directly south → heading ≈ 180°."""
        text = _srt(
            _make_block(frame=1, lat=32.09, lon=34.78),
            _make_block(frame=2, lat=32.08, lon=34.78),
        )
        rows = parse_srt(text, "M1")["M1"]
        self.assertAlmostEqual(float(rows[1]["Heading"]), 180.0, delta=1.0)

    def test_stationary_frame_keeps_previous_heading(self):
        """Identical consecutive positions should not change heading."""
        text = _srt(
            _make_block(frame=1, lat=32.08, lon=34.78),
            _make_block(frame=2, lat=32.09, lon=34.78),   # north → heading ≈ 0
            _make_block(frame=3, lat=32.09, lon=34.78),   # same → heading stays ≈ 0
        )
        rows = parse_srt(text, "M1")["M1"]
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(float(rows[2]["Heading"]), float(rows[1]["Heading"]), places=6)

    def test_heading_field_is_present_in_all_rows(self):
        text = _srt(
            _make_block(frame=1, lat=32.08, lon=34.78),
            _make_block(frame=2, lat=32.09, lon=34.79),
        )
        rows = parse_srt(text, "M1")["M1"]
        for row in rows:
            self.assertIn("Heading", row)
            float(row["Heading"])  # must be numeric string


# ---------------------------------------------------------------------------
# Test: malformed / partial blocks
# ---------------------------------------------------------------------------

class TestMalformedBlocks(unittest.TestCase):

    def test_empty_text_returns_empty_list(self):
        result = parse_srt("", "M1")
        self.assertEqual(result["M1"], [])

    def test_block_without_gps_fields_skipped(self):
        bad_block = "1\n00:00:00,000 --> 00:00:00,016\n<font>no gps here</font>"
        result = parse_srt(bad_block, "M1")
        self.assertEqual(result["M1"], [])

    def test_block_missing_longitude_skipped(self):
        partial = (
            "1\n00:00:00,000 --> 00:00:00,016\n"
            "<font>2025-01-01 00:00:00.000\n"
            "[latitude: 32.08] [rel_alt: 50.0 abs_alt: 45.0] </font>"
        )
        result = parse_srt(partial, "M1")
        self.assertEqual(result["M1"], [])

    def test_block_missing_latitude_skipped(self):
        partial = (
            "1\n00:00:00,000 --> 00:00:00,016\n"
            "<font>2025-01-01 00:00:00.000\n"
            "[longitude: 34.78] [rel_alt: 50.0 abs_alt: 45.0] </font>"
        )
        result = parse_srt(partial, "M1")
        self.assertEqual(result["M1"], [])

    def test_block_missing_rel_alt_skipped(self):
        partial = (
            "1\n00:00:00,000 --> 00:00:00,016\n"
            "<font>2025-01-01 00:00:00.000\n"
            "[latitude: 32.08] [longitude: 34.78] </font>"
        )
        result = parse_srt(partial, "M1")
        self.assertEqual(result["M1"], [])

    def test_mixed_good_and_bad_blocks(self):
        bad_block = "1\n00:00:00,000 --> 00:00:00,016\n<font>garbage data here</font>"
        good_block = _make_block(frame=2, lat=32.08, lon=34.78)
        text = _srt(bad_block, good_block)
        result = parse_srt(text, "M1")
        self.assertEqual(len(result["M1"]), 1)


# ---------------------------------------------------------------------------
# Test: _bearing helper
# ---------------------------------------------------------------------------

class TestBearingHelper(unittest.TestCase):

    def test_bearing_north(self):
        b = _bearing(32.0, 34.78, 32.1, 34.78)
        self.assertAlmostEqual(b, 0.0, delta=0.5)

    def test_bearing_east(self):
        b = _bearing(32.0, 34.78, 32.0, 34.88)
        self.assertAlmostEqual(b, 90.0, delta=0.5)

    def test_bearing_south(self):
        b = _bearing(32.1, 34.78, 32.0, 34.78)
        self.assertAlmostEqual(b, 180.0, delta=0.5)

    def test_bearing_west(self):
        b = _bearing(32.0, 34.88, 32.0, 34.78)
        self.assertAlmostEqual(b, 270.0, delta=0.5)

    def test_bearing_always_in_range(self):
        b = _bearing(32.0, 34.78, 32.1, 34.88)
        self.assertGreaterEqual(b, 0.0)
        self.assertLess(b, 360.0)


# ---------------------------------------------------------------------------
# Test: backend / API compatibility
# ---------------------------------------------------------------------------

class TestBackendCompatibility(unittest.TestCase):
    """
    Verify that parse_srt output is structurally compatible with
    api._missions, covering the exact field access patterns api.py uses.
    """

    def test_output_shape_is_dict_of_list_of_dicts(self):
        text = _srt(_make_block())
        result = parse_srt(text, "TEST")
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["TEST"], list)
        self.assertIsInstance(result["TEST"][0], dict)

    def test_numeric_fields_are_float_parseable(self):
        """api.py calls float(row["Latitude"]) etc. — must not raise."""
        text = _srt(_make_block(lat=32.08, lon=34.78, rel_alt=50.0))
        row = parse_srt(text, "TEST")["TEST"][0]
        self.assertAlmostEqual(float(row["Latitude"]), 32.08)
        self.assertAlmostEqual(float(row["Longitude"]), 34.78)
        self.assertAlmostEqual(float(row["Altitude_m"]), 50.0)

    def test_all_zero_gps_produces_empty_list(self):
        """Pre-flight DJI SRT (all-zero GPS) should yield [], not crash."""
        text = _srt(
            _make_block(lat=0.0, lon=0.0, rel_alt=0.0),
            _make_block(lat=0.0, lon=0.0, rel_alt=0.0),
        )
        result = parse_srt(text, "DJI_0044")
        self.assertEqual(result["DJI_0044"], [])

    def test_srt_upload_via_api(self):
        """POST /upload with a .srt file stores rows in _missions."""
        from fastapi.testclient import TestClient
        import api as api_module
        client = TestClient(api_module.app)

        srt_content = _srt(
            _make_block(frame=1, lat=32.08, lon=34.78, rel_alt=50.0),
            _make_block(frame=2, lat=32.09, lon=34.79, rel_alt=55.0),
        )
        saved = api_module._missions.copy()
        try:
            response = client.post(
                "/upload",
                files={"file": ("DJI_0099.srt", srt_content.encode(), "text/plain")},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("DJI_0099", data["missions"])
            self.assertEqual(data["total_rows"], 2)
        finally:
            api_module._missions = saved  # restore so other tests are unaffected

    def test_srt_upload_wrong_extension_rejected(self):
        """Files with neither .csv nor .srt extension must get 400."""
        from fastapi.testclient import TestClient
        from api import app
        client = TestClient(app)
        response = client.post(
            "/upload",
            files={"file": ("data.txt", b"garbage", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_csv_upload_still_works(self):
        """Extending the endpoint must not break existing CSV uploads."""
        import api as api_module
        from fastapi.testclient import TestClient
        client = TestClient(api_module.app)

        csv_content = (
            "Drone_ID,Mission_ID,Operator_ID,Timestamp,Latitude,Longitude,Altitude_m,"
            "Speed_m_s,Battery_pct,Signal_pct,Mission_Type,Takeoff_Location,"
            "Landing_Location,Crash_Event,GPS_Tampering,Firmware_Version,"
            "Memory_Integrity,Data_Leak_Detected,Weather_Condition,Visibility_km,"
            "Wind_Speed_m_s,Temperature_C,Login_Attempts,Unauthorized_Access,"
            "Data_Encryption_Status,Network_Jamming\n"
            "DRN_0001,TEST_CSV,OP_001,2024-01-01 00:00:00,32.08,34.78,50,10,90,95,"
            "Mapping,TLV,HFA,0,0,v1.0,OK,0,Clear,10,2,20,0,0,None,0\n"
        )
        saved = api_module._missions.copy()
        try:
            response = client.post(
                "/upload",
                files={"file": ("test.csv", csv_content.encode(), "text/csv")},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("TEST_CSV", data["missions"])
        finally:
            api_module._missions.clear()
            api_module._missions.update(saved)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
