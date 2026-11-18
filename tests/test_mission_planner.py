import math
import unittest

from models import GPSCoordinate, CameraConfig, GimbalMode
from mission_planner import MissionPlanner, _scanline_intersections, METRES_PER_LAT_DEG


# ---------------------------------------------------------------------------
# Helper used by tests only: independent ray-cast point-in-polygon check
# ---------------------------------------------------------------------------

def _pip(lat: float, lon: float, polygon: list[GPSCoordinate]) -> bool:
    """Ray-casting PIP — used as an independent verifier in tests."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i].lat, polygon[i].lon
        lat_j, lon_j = polygon[j].lat, polygon[j].lon
        if ((lat_i > lat) != (lat_j > lat)) and (
            lon < (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i) + lon_i
        ):
            inside = not inside
        j = i
    return inside


class TestMissionPlanner(unittest.TestCase):
    # ------------------------------------------------------------------
    # Shared fixtures
    # ------------------------------------------------------------------
    def setUp(self):
        # 100m x 100m square roughly based in Tel Aviv
        self.sw = GPSCoordinate(lat=32.08000, lon=34.78000)
        self.ne = GPSCoordinate(lat=32.08090, lon=34.78106)  # approximately 100m away
        self.polygon = [
            self.sw,
            GPSCoordinate(lat=self.sw.lat, lon=self.ne.lon),
            self.ne,
            GPSCoordinate(lat=self.ne.lat, lon=self.sw.lon),
        ]
        self.camera = CameraConfig(fov_h=90.0, fov_v=90.0, gimbal_mode=GimbalMode.NADIR)

    # ------------------------------------------------------------------
    # Original tests (regression) — must continue to pass
    # ------------------------------------------------------------------

    def test_strip_spacing_no_overlap(self):
        # 100m altitude, 90 deg FOV -> 200m ground footprint width/height
        planner = MissionPlanner(self.polygon, altitude_m=100.0, camera_config=self.camera, overlap_h=0.0, overlap_v=0.0)
        along, cross = planner._compute_strip_spacing_m()
        self.assertAlmostEqual(along, 200.0, places=1)
        self.assertAlmostEqual(cross, 200.0, places=1)

    def test_strip_spacing_50_pct_overlap(self):
        planner = MissionPlanner(self.polygon, altitude_m=100.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        along, cross = planner._compute_strip_spacing_m()
        self.assertAlmostEqual(along, 100.0, places=1)
        self.assertAlmostEqual(cross, 100.0, places=1)

    def test_plan_generates_waypoints(self):
        # Small altitude means small footprint (20x20m), meaning multiple passes needed for 100x100m region
        planner = MissionPlanner(self.polygon, altitude_m=10.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()

        self.assertGreater(len(waypoints), 10, "Should generate multiple waypoints for small footprint over large area")
        # First point should be near SW
        self.assertTrue(waypoints[0].lat >= self.sw.lat)
        self.assertTrue(waypoints[0].lon >= self.sw.lon)

    def test_flight_time(self):
        planner = MissionPlanner(self.polygon, altitude_m=10.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        duration = planner.estimate_flight_time(10.0)
        self.assertGreater(duration, 0.0)

    # ------------------------------------------------------------------
    # New tests: polygon correctness
    # ------------------------------------------------------------------

    def test_no_out_of_polygon_waypoints_triangle(self):
        """All waypoints for a right-triangle polygon must lie inside it."""
        # Right triangle: SW, SE, NE corners (missing NW)
        triangle = [
            GPSCoordinate(lat=32.0800, lon=34.7800),
            GPSCoordinate(lat=32.0800, lon=34.7811),
            GPSCoordinate(lat=32.0809, lon=34.7811),
        ]
        planner = MissionPlanner(triangle, altitude_m=5.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()

        self.assertGreater(len(waypoints), 0, "Triangle should produce at least one waypoint")
        for wp in waypoints:
            self.assertTrue(
                _pip(wp.lat, wp.lon, triangle),
                f"Waypoint ({wp.lat:.6f}, {wp.lon:.6f}) is outside the triangle",
            )

        # The NW bounding-box corner must NOT be present
        for wp in waypoints:
            # In a triangle missing the NW corner, no waypoint should be
            # near lat=max AND lon=min simultaneously
            if wp.lat > 32.0805:
                self.assertGreater(
                    wp.lon, 34.7803,
                    f"Waypoint ({wp.lat:.6f}, {wp.lon:.6f}) appears to be in the NW bounding-box corner",
                )

    def test_no_out_of_polygon_waypoints_lshape(self):
        """Waypoints for an L-shaped polygon must not appear in the missing corner."""
        # L-shape: full square minus top-left quadrant
        # Vertices in order (CCW): BL, BR, TR, right-middle, left-middle, TL-ish
        l_shape = [
            GPSCoordinate(lat=32.0800, lon=34.7800),  # BL
            GPSCoordinate(lat=32.0800, lon=34.7811),  # BR
            GPSCoordinate(lat=32.0809, lon=34.7811),  # TR
            GPSCoordinate(lat=32.0809, lon=34.7806),  # top inner-right
            GPSCoordinate(lat=32.0805, lon=34.7806),  # inner corner
            GPSCoordinate(lat=32.0805, lon=34.7800),  # left inner
        ]
        planner = MissionPlanner(l_shape, altitude_m=5.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()

        self.assertGreater(len(waypoints), 0)
        for wp in waypoints:
            self.assertTrue(
                _pip(wp.lat, wp.lon, l_shape),
                f"Waypoint ({wp.lat:.6f}, {wp.lon:.6f}) is outside the L-shape",
            )

    def test_serpentine_ordering(self):
        """Even strips must go east (increasing lon); odd strips west (decreasing lon)."""
        planner = MissionPlanner(self.polygon, altitude_m=5.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()

        self.assertGreater(len(waypoints), 3)

        # Group waypoints by strip (same lat value)
        strips: dict[float, list[float]] = {}
        for wp in waypoints:
            strips.setdefault(wp.lat, []).append(wp.lon)

        sorted_lats = sorted(strips.keys())
        self.assertGreater(len(sorted_lats), 1, "Need at least 2 strips to check ordering")

        for idx, lat in enumerate(sorted_lats):
            lons = strips[lat]
            if len(lons) < 2:
                continue
            if idx % 2 == 0:
                # Eastbound: longitudes should be non-decreasing
                self.assertEqual(
                    lons, sorted(lons),
                    f"Strip {idx} (lat={lat:.6f}) should be eastbound (increasing lon)",
                )
            else:
                # Westbound: longitudes should be non-increasing
                self.assertEqual(
                    lons, sorted(lons, reverse=True),
                    f"Strip {idx} (lat={lat:.6f}) should be westbound (decreasing lon)",
                )

    def test_no_duplicate_waypoints(self):
        """No two waypoints should share the same (lat, lon) pair."""
        triangle = [
            GPSCoordinate(lat=32.0800, lon=34.7800),
            GPSCoordinate(lat=32.0800, lon=34.7811),
            GPSCoordinate(lat=32.0809, lon=34.7811),
        ]
        planner = MissionPlanner(triangle, altitude_m=5.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()

        coords = [(wp.lat, wp.lon) for wp in waypoints]
        self.assertEqual(len(coords), len(set(coords)), "Duplicate waypoints detected")

    def test_altitude_propagation(self):
        """Every waypoint must carry exactly the requested altitude."""
        requested_alt = 75.0
        planner = MissionPlanner(self.polygon, altitude_m=requested_alt, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()

        self.assertGreater(len(waypoints), 0)
        for wp in waypoints:
            self.assertAlmostEqual(
                wp.alt, requested_alt, places=6,
                msg=f"Waypoint altitude {wp.alt} != requested {requested_alt}",
            )

    def test_degenerate_polygon_raises(self):
        """Three collinear points must raise ValueError on construction."""
        collinear = [
            GPSCoordinate(lat=32.080, lon=34.780),
            GPSCoordinate(lat=32.081, lon=34.780),
            GPSCoordinate(lat=32.082, lon=34.780),
        ]
        with self.assertRaises(ValueError):
            MissionPlanner(collinear, altitude_m=50.0, camera_config=self.camera)

    def test_short_segment_produces_waypoint(self):
        """A polygon that spans only a fraction of a footprint width still yields waypoints."""
        # Very thin but tall sliver: ~2m wide, ~100m tall
        # With 50m altitude + 90° FOV → footprint ~100m wide; strip spacing at 50% overlap = 50m
        # The sliver width (0.00002° lon ≈ 1.8m) is less than half the strip spacing → short-segment path
        sliver = [
            GPSCoordinate(lat=32.0800, lon=34.78000),
            GPSCoordinate(lat=32.0800, lon=34.78002),
            GPSCoordinate(lat=32.0809, lon=34.78002),
            GPSCoordinate(lat=32.0809, lon=34.78000),
        ]
        planner = MissionPlanner(sliver, altitude_m=50.0, camera_config=self.camera, overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()
        self.assertGreater(len(waypoints), 0, "Thin sliver polygon should still produce waypoints")

    def test_scanline_intersections_rectangle(self):
        """_scanline_intersections for a rectangle should return exactly 2 longitudes."""
        rect = [
            GPSCoordinate(lat=32.0800, lon=34.7800),
            GPSCoordinate(lat=32.0800, lon=34.7811),
            GPSCoordinate(lat=32.0809, lon=34.7811),
            GPSCoordinate(lat=32.0809, lon=34.7800),
        ]
        mid_lat = 32.0805
        xs = _scanline_intersections(mid_lat, rect)
        self.assertEqual(len(xs), 2)
        self.assertAlmostEqual(xs[0], 34.7800, places=5)
        self.assertAlmostEqual(xs[1], 34.7811, places=5)

    # ------------------------------------------------------------------
    # New tests: edge coverage completeness
    # ------------------------------------------------------------------

    def test_segment_end_coverage(self):
        """
        For every strip, the last waypoint's right footprint edge must reach
        the polygon's eastern boundary at that latitude (no uncovered eastern tail).
        Uses a polygon whose width is not an exact multiple of delta_lon.
        """
        # 100 x 100 m rectangle — width / delta_lon is not an integer for these params
        planner = MissionPlanner(
            self.polygon, altitude_m=10.0, camera_config=self.camera,
            overlap_h=0.5, overlap_v=0.5,
        )
        waypoints = planner.plan()
        along_m, cross_m = planner._compute_strip_spacing_m()

        lats = [pt.lat for pt in self.polygon]
        mid_lat = (min(lats) + max(lats)) / 2.0
        delta_lon = cross_m / (METRES_PER_LAT_DEG * math.cos(math.radians(mid_lat)))

        # Group waypoints by strip (same lat value, rounded to avoid float noise)
        strips: dict[float, list[float]] = {}
        for wp in waypoints:
            key = round(wp.lat, 9)
            strips.setdefault(key, []).append(wp.lon)

        for lat_key, lons in strips.items():
            xs = _scanline_intersections(lat_key, self.polygon)
            if len(xs) < 2:
                continue
            lon_end = xs[-1]  # rightmost polygon boundary at this latitude
            last_right_edge = max(lons) + delta_lon / 2.0
            self.assertGreaterEqual(
                last_right_edge, lon_end - 1e-8,
                f"Strip lat={lat_key:.7f}: right footprint edge {last_right_edge:.9f} "
                f"< segment end {lon_end:.9f} (gap = {lon_end - last_right_edge:.2e}°)",
            )

    def test_top_edge_strip_coverage(self):
        """
        The northernmost strip's top footprint edge must reach max_lat
        (no uncovered northern band when height is not an exact multiple of delta_lat).
        """
        planner = MissionPlanner(
            self.polygon, altitude_m=10.0, camera_config=self.camera,
            overlap_h=0.5, overlap_v=0.5,
        )
        waypoints = planner.plan()
        along_m, _ = planner._compute_strip_spacing_m()

        lats = [pt.lat for pt in self.polygon]
        max_lat = max(lats)
        delta_lat = along_m / METRES_PER_LAT_DEG

        max_wp_lat = max(wp.lat for wp in waypoints)
        self.assertGreaterEqual(
            max_wp_lat + delta_lat / 2.0, max_lat - 1e-8,
            f"Northern strip top edge {max_wp_lat + delta_lat / 2.0:.9f} "
            f"< max_lat {max_lat:.9f}",
        )

    def test_non_exact_multiple_dimensions(self):
        """
        For a polygon whose height AND width are intentionally not exact multiples
        of the strip/waypoint spacing, verify both eastern and northern edge coverage.
        """
        # Design a rectangle with non-integer multiples:
        # altitude=7m, 90° FOV → footprint=14m, spacing at 50% overlap = 7m
        # delta_lat ≈ 7/111320 ≈ 6.29e-5°
        # delta_lon ≈ 7/(111320*cos(32°)) ≈ 7.42e-5°
        # polygon height = 17 * delta_lat * 1.4 → non-multiple
        # polygon width  = 17 * delta_lon * 1.6 → non-multiple
        altitude = 7.0
        camera = CameraConfig(fov_h=90.0, fov_v=90.0, gimbal_mode=GimbalMode.NADIR)
        along_m = 2.0 * altitude * math.tan(math.radians(45.0)) * 0.5  # 7 m
        cross_m = along_m

        base_lat = 32.0800
        base_lon = 34.7800
        cos_lat = math.cos(math.radians(base_lat))
        delta_lat = along_m / METRES_PER_LAT_DEG
        delta_lon = cross_m / (METRES_PER_LAT_DEG * cos_lat)

        # Height = 8.7 strips, Width = 5.3 strips — both non-integer
        poly_h = delta_lat * 8.7
        poly_w = delta_lon * 5.3
        polygon = [
            GPSCoordinate(lat=base_lat,           lon=base_lon),
            GPSCoordinate(lat=base_lat,           lon=base_lon + poly_w),
            GPSCoordinate(lat=base_lat + poly_h,  lon=base_lon + poly_w),
            GPSCoordinate(lat=base_lat + poly_h,  lon=base_lon),
        ]
        max_lat = base_lat + poly_h

        planner = MissionPlanner(polygon, altitude_m=altitude, camera_config=camera,
                                 overlap_h=0.5, overlap_v=0.5)
        waypoints = planner.plan()
        self.assertGreater(len(waypoints), 0)

        # --- northern edge ---
        max_wp_lat = max(wp.lat for wp in waypoints)
        self.assertGreaterEqual(
            max_wp_lat + delta_lat / 2.0, max_lat - 1e-8,
            "Northern band uncovered: top footprint edge falls short of max_lat",
        )

        # --- eastern edge per strip ---
        strips: dict[float, list[float]] = {}
        for wp in waypoints:
            key = round(wp.lat, 9)
            strips.setdefault(key, []).append(wp.lon)

        for lat_key, lons in strips.items():
            xs = _scanline_intersections(lat_key, polygon)
            if len(xs) < 2:
                continue
            lon_end = xs[-1]
            last_right_edge = max(lons) + delta_lon / 2.0
            self.assertGreaterEqual(
                last_right_edge, lon_end - 1e-8,
                f"Eastern tail uncovered at lat={lat_key:.7f}",
            )


    def test_small_height_polygon_gets_waypoints(self):
        """
        A polygon whose north-south height is less than delta_lat / 2 must still
        produce at least one waypoint (the regular strip grid would otherwise
        yield nothing because the first candidate strip falls above max_lat).
        """
        along_m, _ = MissionPlanner(
            self.polygon, altitude_m=100.0, camera_config=self.camera,
            overlap_h=0.5, overlap_v=0.5,
        )._compute_strip_spacing_m()
        delta_lat = along_m / METRES_PER_LAT_DEG

        # Polygon height = delta_lat / 4  (well below the half-step threshold)
        tiny_h = delta_lat / 4.0
        base_lat = 32.0800
        base_lon = 34.7800
        tiny_polygon = [
            GPSCoordinate(lat=base_lat,            lon=base_lon),
            GPSCoordinate(lat=base_lat,            lon=base_lon + 0.001),
            GPSCoordinate(lat=base_lat + tiny_h,   lon=base_lon + 0.001),
            GPSCoordinate(lat=base_lat + tiny_h,   lon=base_lon),
        ]
        planner = MissionPlanner(
            tiny_polygon, altitude_m=100.0, camera_config=self.camera,
            overlap_h=0.5, overlap_v=0.5,
        )
        waypoints = planner.plan()
        self.assertGreater(len(waypoints), 0, "Small-height polygon produced no waypoints")
        # The single strip should be inside the polygon's lat range
        mid_lat = base_lat + tiny_h / 2.0
        for wp in waypoints:
            self.assertGreaterEqual(wp.lat, base_lat)
            self.assertLessEqual(wp.lat, base_lat + tiny_h + delta_lat)


if __name__ == "__main__":
    unittest.main()
