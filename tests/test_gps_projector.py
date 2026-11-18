"""
test_gps_projector.py  —  Unit tests for Task 4: GPSProjector
==============================================================
Run with:
    python -m pytest test_gps_projector.py -v
"""

import unittest
import numpy as np

from models import GPSCoordinate
from drone_oversight import MosaicResult, GPSProjector


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

# A small area (roughly 1 km × 0.5 km) over Tel Aviv
_SW_LAT, _SW_LON = 32.08, 34.78   # bottom-left (south-west)
_NE_LAT, _NE_LON = 32.09, 34.79   # top-right   (north-east)

# Use odd pixel dimensions so the exact centre pixel is integer-valued.
_W, _H = 101, 51


def _make_mosaic(
    w: int = _W,
    h: int = _H,
    sw_lat: float = _SW_LAT,
    sw_lon: float = _SW_LON,
    ne_lat: float = _NE_LAT,
    ne_lon: float = _NE_LON,
) -> MosaicResult:
    """Create a minimal north-up MosaicResult for testing."""
    image = np.zeros((h, w, 3), dtype=np.uint8)
    bounding_box = [
        GPSCoordinate(lat=ne_lat, lon=sw_lon),   # TL: NW corner
        GPSCoordinate(lat=ne_lat, lon=ne_lon),   # TR: NE corner
        GPSCoordinate(lat=sw_lat, lon=ne_lon),   # BR: SE corner
        GPSCoordinate(lat=sw_lat, lon=sw_lon),   # BL: SW corner
    ]
    return MosaicResult(image=image, bounding_box=bounding_box, gsd_m=0.1)


# ---------------------------------------------------------------------------
# pixel_to_gps tests
# ---------------------------------------------------------------------------

class TestPixelToGPS(unittest.TestCase):

    def setUp(self):
        self.mosaic = _make_mosaic()
        self.proj   = GPSProjector(self.mosaic)

    def test_top_left_pixel_is_nw_corner(self):
        """Pixel (0, 0) must map to the north-west GPS corner."""
        coord = self.proj.pixel_to_gps(0, 0)
        self.assertAlmostEqual(coord.lat, _NE_LAT, places=10)
        self.assertAlmostEqual(coord.lon, _SW_LON, places=10)

    def test_top_right_pixel_is_ne_corner(self):
        """Pixel (W-1, 0) must map to the north-east GPS corner."""
        coord = self.proj.pixel_to_gps(_W - 1, 0)
        self.assertAlmostEqual(coord.lat, _NE_LAT, places=10)
        self.assertAlmostEqual(coord.lon, _NE_LON, places=10)

    def test_bottom_right_pixel_is_se_corner(self):
        """Pixel (W-1, H-1) must map to the south-east GPS corner."""
        coord = self.proj.pixel_to_gps(_W - 1, _H - 1)
        self.assertAlmostEqual(coord.lat, _SW_LAT, places=10)
        self.assertAlmostEqual(coord.lon, _NE_LON, places=10)

    def test_bottom_left_pixel_is_sw_corner(self):
        """Pixel (0, H-1) must map to the south-west GPS corner."""
        coord = self.proj.pixel_to_gps(0, _H - 1)
        self.assertAlmostEqual(coord.lat, _SW_LAT, places=10)
        self.assertAlmostEqual(coord.lon, _SW_LON, places=10)

    def test_center_pixel_is_center_gps(self):
        """Centre pixel must map to the geographic centre of the mosaic."""
        coord = self.proj.pixel_to_gps(_W // 2, _H // 2)
        expected_lat = (_NE_LAT + _SW_LAT) / 2.0
        expected_lon = (_NE_LON + _SW_LON) / 2.0
        self.assertAlmostEqual(coord.lat, expected_lat, places=10)
        self.assertAlmostEqual(coord.lon, expected_lon, places=10)

    def test_returns_gps_coordinate(self):
        coord = self.proj.pixel_to_gps(10, 10)
        self.assertIsInstance(coord, GPSCoordinate)

    def test_negative_pixel_x_raises(self):
        with self.assertRaises(IndexError):
            self.proj.pixel_to_gps(-1, 0)

    def test_negative_pixel_y_raises(self):
        with self.assertRaises(IndexError):
            self.proj.pixel_to_gps(0, -1)

    def test_pixel_x_at_width_raises(self):
        with self.assertRaises(IndexError):
            self.proj.pixel_to_gps(_W, 0)

    def test_pixel_y_at_height_raises(self):
        with self.assertRaises(IndexError):
            self.proj.pixel_to_gps(0, _H)


# ---------------------------------------------------------------------------
# gps_to_pixel tests
# ---------------------------------------------------------------------------

class TestGPSToPixel(unittest.TestCase):

    def setUp(self):
        self.mosaic = _make_mosaic()
        self.proj   = GPSProjector(self.mosaic)

    def test_nw_corner_is_top_left_pixel(self):
        px, py = self.proj.gps_to_pixel(GPSCoordinate(lat=_NE_LAT, lon=_SW_LON))
        self.assertEqual(px, 0)
        self.assertEqual(py, 0)

    def test_ne_corner_is_top_right_pixel(self):
        px, py = self.proj.gps_to_pixel(GPSCoordinate(lat=_NE_LAT, lon=_NE_LON))
        self.assertEqual(px, _W - 1)
        self.assertEqual(py, 0)

    def test_se_corner_is_bottom_right_pixel(self):
        px, py = self.proj.gps_to_pixel(GPSCoordinate(lat=_SW_LAT, lon=_NE_LON))
        self.assertEqual(px, _W - 1)
        self.assertEqual(py, _H - 1)

    def test_sw_corner_is_bottom_left_pixel(self):
        px, py = self.proj.gps_to_pixel(GPSCoordinate(lat=_SW_LAT, lon=_SW_LON))
        self.assertEqual(px, 0)
        self.assertEqual(py, _H - 1)

    def test_center_gps_is_center_pixel(self):
        center = GPSCoordinate(
            lat=(_NE_LAT + _SW_LAT) / 2.0,
            lon=(_NE_LON + _SW_LON) / 2.0,
        )
        px, py = self.proj.gps_to_pixel(center)
        self.assertEqual(px, _W // 2)
        self.assertEqual(py, _H // 2)

    def test_gps_outside_lat_raises(self):
        with self.assertRaises(ValueError):
            self.proj.gps_to_pixel(GPSCoordinate(lat=90.0, lon=_SW_LON))

    def test_gps_outside_lon_raises(self):
        with self.assertRaises(ValueError):
            self.proj.gps_to_pixel(GPSCoordinate(lat=_SW_LAT, lon=0.0))

    def test_returns_tuple_of_ints(self):
        result = self.proj.gps_to_pixel(GPSCoordinate(lat=_SW_LAT, lon=_SW_LON))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):

    def setUp(self):
        self.mosaic = _make_mosaic()
        self.proj   = GPSProjector(self.mosaic)

    def test_pixel_to_gps_to_pixel_corners(self):
        """pixel → GPS → pixel must recover the original pixel for all corners."""
        for px_in, py_in in [(0, 0), (_W - 1, 0), (_W - 1, _H - 1), (0, _H - 1)]:
            coord = self.proj.pixel_to_gps(px_in, py_in)
            px_out, py_out = self.proj.gps_to_pixel(coord)
            self.assertEqual(px_out, px_in, msg=f"pixel_x mismatch for ({px_in}, {py_in})")
            self.assertEqual(py_out, py_in, msg=f"pixel_y mismatch for ({px_in}, {py_in})")

    def test_pixel_to_gps_to_pixel_center(self):
        """Round-trip at the centre pixel."""
        px_in, py_in = _W // 2, _H // 2
        coord = self.proj.pixel_to_gps(px_in, py_in)
        px_out, py_out = self.proj.gps_to_pixel(coord)
        self.assertEqual(px_out, px_in)
        self.assertEqual(py_out, py_in)

    def test_gps_to_pixel_to_gps_center(self):
        """GPS → pixel → GPS round-trip at the geographic centre."""
        lat_in = (_NE_LAT + _SW_LAT) / 2.0
        lon_in = (_NE_LON + _SW_LON) / 2.0
        coord_in = GPSCoordinate(lat=lat_in, lon=lon_in)
        px, py = self.proj.gps_to_pixel(coord_in)
        coord_out = self.proj.pixel_to_gps(px, py)
        self.assertAlmostEqual(coord_out.lat, lat_in, places=10)
        self.assertAlmostEqual(coord_out.lon, lon_in, places=10)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
