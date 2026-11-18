"""
test_heatmap.py  —  Unit tests for Task 2: CoverageHeatmap
============================================================
Run with:
    python test_heatmap.py
or:
    python -m pytest test_heatmap.py -v
"""

import math
import unittest

from models import GPSCoordinate, CameraConfig, DroneTelemetry, GimbalMode
from footprint_calculator import FootprintCalculator, FootprintResult
from coverage_heatmap import CoverageHeatmap

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# A 500 m × 500 m AOI over Tel Aviv
_SW = GPSCoordinate(lat=32.0800, lon=34.7780)
_NE = GPSCoordinate(lat=32.0845, lon=34.7836)

def _make_area() -> list[GPSCoordinate]:
    return [
        _SW,
        GPSCoordinate(lat=_SW.lat, lon=_NE.lon),
        _NE,
        GPSCoordinate(lat=_NE.lat, lon=_SW.lon),
    ]

def _make_footprint(lat: float, lon: float, alt: float,
                    fov_h: float = 90.0, fov_v: float = 60.0,
                    heading: float = 0.0) -> FootprintResult:
    """Helper: create a nadir footprint centred at (lat, lon, alt)."""
    tel = DroneTelemetry(
        position=GPSCoordinate(lat=lat, lon=lon, alt=alt),
        heading=heading,
        camera=CameraConfig(fov_h=fov_h, fov_v=fov_v, gimbal_mode=GimbalMode.NADIR),
    )
    return FootprintCalculator(tel).compute()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestCoverageHeatmapConstruction(unittest.TestCase):
    """Test grid is built correctly from area corners."""

    def test_grid_dimensions_positive(self):
        hm = CoverageHeatmap(_make_area(), grid_resolution_m=10.0)
        rows, cols = hm.shape
        self.assertGreater(rows, 0)
        self.assertGreater(cols, 0)

    def test_initial_coverage_zero(self):
        hm = CoverageHeatmap(_make_area(), grid_resolution_m=10.0)
        self.assertEqual(hm.coverage_percentage(), 0.0)

    def test_invalid_resolution_raises(self):
        with self.assertRaises(ValueError):
            CoverageHeatmap(_make_area(), grid_resolution_m=0.0)

    def test_too_few_corners_raises(self):
        with self.assertRaises(ValueError):
            CoverageHeatmap([_SW, _NE], grid_resolution_m=10.0)


class TestAddFootprint(unittest.TestCase):
    """Test footprint registration and hit-count logic."""

    def test_full_coverage(self):
        """A footprint much larger than the AOI must cover 100 % of cells."""
        hm = CoverageHeatmap(_make_area(), grid_resolution_m=10.0)
        # Huge footprint: 1 km altitude → ~2000 m × 1155 m footprint
        fp = _make_footprint(lat=32.0822, lon=34.7808, alt=1000.0,
                             fov_h=90.0, fov_v=60.0)
        hm.add_footprint(fp)
        self.assertAlmostEqual(hm.coverage_percentage(), 100.0, delta=1.0)

    def test_partial_coverage(self):
        """A small footprint at the centre covers less than 100 % of the AOI."""
        hm = CoverageHeatmap(_make_area(), grid_resolution_m=10.0)
        # Tiny footprint: 5 m altitude → very small area
        fp = _make_footprint(lat=32.0822, lon=34.7808, alt=5.0,
                             fov_h=30.0, fov_v=20.0)
        hm.add_footprint(fp)
        pct = hm.coverage_percentage()
        self.assertGreater(pct, 0.0)
        self.assertLess(pct, 100.0)

    def test_overlap_counts(self):
        """Cells in the overlap zone of two footprints must have count >= 2."""
        hm = CoverageHeatmap(_make_area(), grid_resolution_m=10.0)
        # Both footprints centred at the same point → complete overlap
        fp1 = _make_footprint(lat=32.0822, lon=34.7808, alt=80.0)
        fp2 = _make_footprint(lat=32.0822, lon=34.7808, alt=80.0)
        hm.add_footprint(fp1)
        hm.add_footprint(fp2)
        # Every covered cell should have count == 2
        grid = hm.get_grid()
        self.assertEqual(int(grid.max()), 2)
        self.assertEqual(hm.max_overlap(), 2)

    def test_footprint_count(self):
        hm = CoverageHeatmap(_make_area(), grid_resolution_m=10.0)
        hm.add_footprint(_make_footprint(32.0822, 34.7808, 80.0))
        hm.add_footprint(_make_footprint(32.0822, 34.7825, 80.0))
        self.assertEqual(hm.num_footprints, 2)


class TestQueryMethods(unittest.TestCase):
    """Test get_cell_center, covered_gps_points, uncovered_gps_points."""

    def setUp(self):
        self.hm = CoverageHeatmap(_make_area(), grid_resolution_m=10.0)
        fp = _make_footprint(lat=32.0822, lon=34.7808, alt=80.0)
        self.hm.add_footprint(fp)

    def test_get_cell_center_in_bounds(self):
        center = self.hm.get_cell_center(0, 0)
        self.assertIsInstance(center, GPSCoordinate)

    def test_get_cell_center_out_of_bounds(self):
        rows, cols = self.hm.shape
        with self.assertRaises(IndexError):
            self.hm.get_cell_center(rows + 10, cols + 10)

    def test_covered_gps_points_non_empty(self):
        points = list(self.hm.covered_gps_points())
        self.assertGreater(len(points), 0)
        self.assertIsInstance(points[0], GPSCoordinate)

    def test_uncovered_gps_points_non_empty(self):
        points = list(self.hm.uncovered_gps_points())
        # The footprint does NOT cover the entire AOI, so there are uncovered cells
        self.assertGreater(len(points), 0)

    def test_covered_plus_uncovered_equals_total(self):
        rows, cols = self.hm.shape
        total = rows * cols
        covered   = len(list(self.hm.covered_gps_points()))
        uncovered = len(list(self.hm.uncovered_gps_points()))
        self.assertEqual(covered + uncovered, total)


# ---------------------------------------------------------------------------
# Task 1 — Oblique footprint tests
# ---------------------------------------------------------------------------

def _make_oblique_footprint(
    lat: float, lon: float, alt: float,
    pitch: float,
    fov_h: float = 90.0, fov_v: float = 60.0,
    heading: float = 0.0,
) -> FootprintResult:
    """Helper: create an oblique footprint centred at (lat, lon, alt)."""
    tel = DroneTelemetry(
        position=GPSCoordinate(lat=lat, lon=lon, alt=alt),
        heading=heading,
        camera=CameraConfig(
            fov_h=fov_h, fov_v=fov_v,
            gimbal_mode=GimbalMode.OBLIQUE,
            gimbal_pitch=pitch,
        ),
    )
    return FootprintCalculator(tel).compute()


class TestObliqueFootprint(unittest.TestCase):
    """Task 1 — oblique gimbal mode."""

    # Drone position used across tests
    LAT, LON, ALT = 32.0822, 34.7808, 100.0

    def test_zero_pitch_matches_nadir_width(self):
        """pitch=0° oblique footprint must have same width as nadir."""
        nadir_fp  = _make_footprint(self.LAT, self.LON, self.ALT)
        obliq_fp  = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=0.0)
        self.assertAlmostEqual(nadir_fp.ground_width_m, obliq_fp.ground_width_m, places=4)

    def test_zero_pitch_matches_nadir_height(self):
        """pitch=0° oblique footprint must have same height as nadir."""
        nadir_fp  = _make_footprint(self.LAT, self.LON, self.ALT, fov_v=60.0)
        obliq_fp  = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=0.0, fov_v=60.0)
        self.assertAlmostEqual(nadir_fp.ground_height_m, obliq_fp.ground_height_m, places=4)

    def test_forward_pitch_shifts_footprint_north(self):
        """Heading=0° (North), pitch>0° → footprint centre should be north of nadir."""
        fp = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=20.0, heading=0.0)
        # Average lat of footprint corners
        avg_lat = sum(c.lat for c in fp.corners) / 4
        self.assertGreater(avg_lat, self.LAT)

    def test_width_constant_with_pitch(self):
        """Width should not change as pitch increases (cross-heading axis unaffected)."""
        fp0  = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=0.0)
        fp20 = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=20.0)
        self.assertAlmostEqual(fp0.ground_width_m, fp20.ground_width_m, places=4)

    def test_height_increases_with_pitch(self):
        """Along-heading extent grows as pitch increases (asymmetric near/far rays)."""
        fp0  = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=0.0,  fov_v=60.0)
        fp20 = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=20.0, fov_v=60.0)
        self.assertGreater(fp20.ground_height_m, fp0.ground_height_m)

    def test_near_far_distances_correct(self):
        """pitch=30°, fov_v=60° → near_dist=H*tan(0°)=0, far_dist=H*tan(60°)."""
        H, pitch, fov_v = 100.0, 30.0, 60.0
        fp = _make_oblique_footprint(self.LAT, self.LON, H, pitch=pitch, fov_v=fov_v, heading=0.0)
        expected_near = H * math.tan(math.radians(pitch - fov_v / 2))  # tan(0°) = 0
        expected_far  = H * math.tan(math.radians(pitch + fov_v / 2))  # tan(60°) = √3
        expected_height = expected_far - expected_near
        self.assertAlmostEqual(fp.ground_height_m, expected_height, places=2)

    def test_horizon_pitch_raises(self):
        """pitch + fov_v/2 >= 90° must raise ValueError."""
        with self.assertRaises(ValueError):
            _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=60.0, fov_v=60.0)

    def test_exactly_at_horizon_raises(self):
        """pitch + fov_v/2 == 90° exactly must also raise."""
        with self.assertRaises(ValueError):
            _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=60.0, fov_v=60.0)

    def test_nadir_is_drone_position(self):
        """Nadir point must always be directly below the drone regardless of pitch."""
        fp = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=20.0)
        self.assertAlmostEqual(fp.nadir.lat, self.LAT, places=6)
        self.assertAlmostEqual(fp.nadir.lon, self.LON, places=6)

    def test_returns_footprint_result(self):
        fp = _make_oblique_footprint(self.LAT, self.LON, self.ALT, pitch=15.0)
        self.assertIsInstance(fp, FootprintResult)
        self.assertEqual(len(fp.corners), 4)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
