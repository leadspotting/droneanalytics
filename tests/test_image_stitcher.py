"""
test_image_stitcher.py  —  Unit tests for Task 3: ImageStitcher
===============================================================
Run with:
    python -m pytest test_image_stitcher.py -v
"""

import math
import unittest

import numpy as np

from models import GPSCoordinate, CameraConfig, GimbalMode, DroneTelemetry
from footprint_calculator import FootprintCalculator, FootprintResult
from drone_oversight import ImageStitcher, MosaicResult, _dlt_homography


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_CAMERA = CameraConfig(fov_h=90.0, fov_v=60.0, gimbal_mode=GimbalMode.NADIR)

# At altitude=100m, fov_h=90°: ground_width = 2*100*tan(45°) = 200 m
# At altitude=100m, fov_v=60°: ground_height = 2*100*tan(30°) ≈ 115.47 m
_ALT   = 100.0
_LAT   = 32.0
_LON   = 34.0
_GSD   = 1.0   # metres/pixel → image is 200×115 px


def _make_footprint(
    lat: float = _LAT,
    lon: float = _LON,
    alt: float = _ALT,
    heading: float = 0.0,
    camera: CameraConfig = _CAMERA,
) -> FootprintResult:
    tel = DroneTelemetry(
        position=GPSCoordinate(lat=lat, lon=lon, alt=alt),
        heading=heading,
        camera=camera,
    )
    return FootprintCalculator(tel).compute()


def _image_for_footprint(
    fp: FootprintResult,
    color: tuple[int, int, int] = (128, 64, 32),
    gsd: float = _GSD,
) -> np.ndarray:
    """Create a solid-colour image whose pixel dimensions match the footprint at the given GSD."""
    W = max(1, round(fp.ground_width_m / gsd))
    H = max(1, round(fp.ground_height_m / gsd))
    img = np.full((H, W, 3), color, dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# Test: DLT homography helper
# ---------------------------------------------------------------------------

class TestDLTHomography(unittest.TestCase):

    def test_identity_mapping(self):
        """Same src and dst points → homography ≈ identity."""
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        H = _dlt_homography(pts, pts)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-8)

    def test_pure_translation(self):
        """Shift all points by (0.5, 0.5) → translation matrix."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        dst = src + np.array([0.5, 0.5])
        H = _dlt_homography(src, dst)
        # Apply H to each src point and check it gives dst
        for s, d in zip(src, dst):
            sh = H @ np.array([s[0], s[1], 1.0])
            computed = sh[:2] / sh[2]
            np.testing.assert_allclose(computed, d, atol=1e-8)

    def test_uniform_scale(self):
        """Scale by 2 → H[0,0] = H[1,1] ≈ 2 (after normalising H[2,2]=1)."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        dst = src * 2.0
        H = _dlt_homography(src, dst)
        for s, d in zip(src, dst):
            sh = H @ np.array([s[0], s[1], 1.0])
            computed = sh[:2] / sh[2]
            np.testing.assert_allclose(computed, d, atol=1e-8)

    def test_returns_3x3(self):
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        H = _dlt_homography(src, src)
        self.assertEqual(H.shape, (3, 3))

    def test_normalised_H22_is_one(self):
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        dst = src + 0.1
        H = _dlt_homography(src, dst)
        self.assertAlmostEqual(H[2, 2], 1.0, places=10)


# ---------------------------------------------------------------------------
# Test: _compute_homography
# ---------------------------------------------------------------------------

class TestComputeHomography(unittest.TestCase):

    def setUp(self):
        self.fp = _make_footprint()
        self.img = _image_for_footprint(self.fp)
        self.stitcher = ImageStitcher([self.img], [self.fp], _CAMERA)

    def test_same_footprint_gives_identity(self):
        """_compute_homography(fp, fp) must return identity."""
        H = self.stitcher._compute_homography(self.fp, self.fp)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-8)

    def test_returns_3x3_ndarray(self):
        H = self.stitcher._compute_homography(self.fp, self.fp)
        self.assertIsInstance(H, np.ndarray)
        self.assertEqual(H.shape, (3, 3))

    def test_larger_dst_maps_src_corners_inward(self):
        """Src footprint inside a 2× larger dst: src corners map into (0.25, 0.75)."""
        fp_small = _make_footprint(alt=100.0)   # ground_width = 200 m
        fp_large = _make_footprint(alt=200.0)   # ground_width = 400 m, same centre
        H = self.stitcher._compute_homography(fp_small, fp_large)

        # TL of src (normalised 0,0) should land inside dst, near centre-left
        pt_src = np.array([0.0, 0.0, 1.0])
        pt_dst_h = H @ pt_src
        pt_dst = pt_dst_h[:2] / pt_dst_h[2]
        # Should be between 0 and 0.5 in both axes (not at the corner)
        self.assertGreater(pt_dst[0], 0.0)
        self.assertLess(pt_dst[0], 0.5)
        self.assertGreater(pt_dst[1], 0.0)
        self.assertLess(pt_dst[1], 0.5)

    def test_src_corners_map_within_unit_square_when_contained(self):
        """When src is smaller than dst, all mapped corners must fall in [0,1]."""
        fp_small = _make_footprint(alt=100.0)
        fp_large = _make_footprint(alt=200.0)
        H = self.stitcher._compute_homography(fp_small, fp_large)

        src_corners = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
        for sc in src_corners:
            ph = H @ np.array([sc[0], sc[1], 1.0])
            p = ph[:2] / ph[2]
            self.assertGreaterEqual(p[0], 0.0 - 1e-9)
            self.assertLessEqual(p[0],    1.0 + 1e-9)
            self.assertGreaterEqual(p[1], 0.0 - 1e-9)
            self.assertLessEqual(p[1],    1.0 + 1e-9)


# ---------------------------------------------------------------------------
# Test: stitch — single image
# ---------------------------------------------------------------------------

class TestStitchSingleImage(unittest.TestCase):

    def setUp(self):
        self.fp = _make_footprint()
        self.color = (200, 100, 50)
        self.img = _image_for_footprint(self.fp, color=self.color, gsd=_GSD)
        self.stitcher = ImageStitcher([self.img], [self.fp], _CAMERA)
        self.result = self.stitcher.stitch()

    def test_returns_mosaic_result(self):
        self.assertIsInstance(self.result, MosaicResult)

    def test_image_is_3channel_uint8(self):
        self.assertEqual(self.result.image.dtype, np.uint8)
        self.assertEqual(self.result.image.ndim, 3)
        self.assertEqual(self.result.image.shape[2], 3)

    def test_canvas_size_matches_footprint_at_gsd(self):
        """Canvas W and H should equal source image dimensions for a single image at default GSD."""
        H_img, W_img = self.img.shape[:2]
        H_canvas, W_canvas = self.result.image.shape[:2]
        # Allow ±1 pixel from floating-point rounding
        self.assertAlmostEqual(H_canvas, H_img, delta=1)
        self.assertAlmostEqual(W_canvas, W_img, delta=1)

    def test_bounding_box_has_four_corners(self):
        self.assertEqual(len(self.result.bounding_box), 4)
        for c in self.result.bounding_box:
            self.assertIsInstance(c, GPSCoordinate)

    def test_bounding_box_nw_corner_is_top_left(self):
        """bounding_box[0] must be the NW (max lat, min lon) corner."""
        bb = self.result.bounding_box
        fp_lats = [c.lat for c in self.fp.corners]
        fp_lons = [c.lon for c in self.fp.corners]
        self.assertAlmostEqual(bb[0].lat, max(fp_lats), places=10)
        self.assertAlmostEqual(bb[0].lon, min(fp_lons), places=10)

    def test_bounding_box_se_corner_is_bottom_right(self):
        """bounding_box[2] must be the SE (min lat, max lon) corner."""
        bb = self.result.bounding_box
        fp_lats = [c.lat for c in self.fp.corners]
        fp_lons = [c.lon for c in self.fp.corners]
        self.assertAlmostEqual(bb[2].lat, min(fp_lats), places=10)
        self.assertAlmostEqual(bb[2].lon, max(fp_lons), places=10)

    def test_gsd_is_positive(self):
        self.assertGreater(self.result.gsd_m, 0.0)

    def test_gsd_matches_footprint_and_image_width(self):
        """GSD should equal ground_width_m / image_width_px."""
        expected_gsd = self.fp.ground_width_m / self.img.shape[1]
        self.assertAlmostEqual(self.result.gsd_m, expected_gsd, places=8)

    def test_content_preserved_for_solid_colour(self):
        """Single solid-colour source → canvas should contain only that colour."""
        canvas = self.result.image
        # At least 90 % of pixels should be the source colour
        # (Border rounding may leave a thin strip unpainted)
        color_arr = np.array(self.color, dtype=np.uint8)
        matches = np.all(canvas == color_arr, axis=2)
        pct_painted = matches.sum() / matches.size
        self.assertGreater(pct_painted, 0.90,
                           msg=f"Only {pct_painted:.1%} of canvas pixels have the expected colour")


# ---------------------------------------------------------------------------
# Test: stitch — multiple images
# ---------------------------------------------------------------------------

class TestStitchMultipleImages(unittest.TestCase):

    def _make_adjacent_footprints(self):
        """
        Two footprints that sit side by side (no overlap) along longitude.
        The second footprint is placed so its western edge touches the eastern
        edge of the first.
        """
        fp1 = _make_footprint(lat=_LAT, lon=_LON, alt=_ALT)
        # Move the second footprint one full ground_width east
        METRES_PER_LAT_DEG = 111_320.0
        avg_lat_rad = math.radians(_LAT)
        metres_per_lon = METRES_PER_LAT_DEG * math.cos(avg_lat_rad)
        lon_step_deg = fp1.ground_width_m / metres_per_lon
        fp2 = _make_footprint(lat=_LAT, lon=_LON + lon_step_deg, alt=_ALT)
        return fp1, fp2

    def test_two_images_returns_mosaic_result(self):
        fp1, fp2 = self._make_adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        stitcher = ImageStitcher([img1, img2], [fp1, fp2], _CAMERA)
        result = stitcher.stitch()
        self.assertIsInstance(result, MosaicResult)

    def test_overlapping_images_are_blended_not_overwritten(self):
        fp1 = _make_footprint(lat=_LAT, lon=_LON, alt=_ALT)
        fp2 = _make_footprint(lat=_LAT, lon=_LON, alt=_ALT)
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 0, 255))
        result = ImageStitcher([img1, img2], [fp1, fp2], _CAMERA).stitch()
        centre = result.image[result.image.shape[0] // 2, result.image.shape[1] // 2]
        self.assertGreater(centre[0], 0)
        self.assertGreater(centre[2], 0)
        self.assertLess(centre[0], 255)
        self.assertLess(centre[2], 255)

    def test_canvas_covers_both_footprints(self):
        """Canvas bounding box must contain GPS corners of both footprints."""
        fp1, fp2 = self._make_adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        stitcher = ImageStitcher([img1, img2], [fp1, fp2], _CAMERA)
        result = stitcher.stitch()

        all_lats = [c.lat for fp in (fp1, fp2) for c in fp.corners]
        all_lons = [c.lon for fp in (fp1, fp2) for c in fp.corners]
        bb_lats = [c.lat for c in result.bounding_box]
        bb_lons = [c.lon for c in result.bounding_box]

        self.assertAlmostEqual(max(bb_lats), max(all_lats), places=10)
        self.assertAlmostEqual(min(bb_lats), min(all_lats), places=10)
        self.assertAlmostEqual(max(bb_lons), max(all_lons), places=10)
        self.assertAlmostEqual(min(bb_lons), min(all_lons), places=10)

    def test_canvas_wider_than_single_image(self):
        """Two side-by-side images → canvas width > single image width."""
        fp1, fp2 = self._make_adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))

        single = ImageStitcher([img1], [fp1], _CAMERA).stitch()
        combined = ImageStitcher([img1, img2], [fp1, fp2], _CAMERA).stitch()

        self.assertGreater(combined.image.shape[1], single.image.shape[1])

    def test_gsd_uses_finest_resolution(self):
        """GSD must equal the minimum per-image GSD (finest resolution)."""
        fp1, fp2 = self._make_adjacent_footprints()
        img1 = _image_for_footprint(fp1, gsd=1.0)    # gsd=1.0
        img2 = _image_for_footprint(fp2, gsd=2.0)    # coarser → image half as wide
        # Halve img2's width so its GSD = 2 m/px
        img2_half = img2[:, ::2, :].copy()   # still stride trick; recalculate properly
        img2_half = np.full(
            (max(1, round(fp2.ground_height_m / 2.0)),
             max(1, round(fp2.ground_width_m / 2.0)),
             3),
            (0, 200, 0), dtype=np.uint8
        )

        stitcher = ImageStitcher([img1, img2_half], [fp1, fp2], _CAMERA)
        result = stitcher.stitch()

        gsd1 = fp1.ground_width_m / img1.shape[1]     # 1.0 m/px
        gsd2 = fp2.ground_width_m / img2_half.shape[1]  # 2.0 m/px
        expected_gsd = min(gsd1, gsd2)
        self.assertAlmostEqual(result.gsd_m, expected_gsd, places=6)

    def test_overlapping_images_no_crash(self):
        """Two identical footprints (full overlap) must not raise."""
        fp = _make_footprint()
        img1 = _image_for_footprint(fp, color=(200, 0, 0))
        img2 = _image_for_footprint(fp, color=(0, 200, 0))
        stitcher = ImageStitcher([img1, img2], [fp, fp], _CAMERA)
        result = stitcher.stitch()
        self.assertIsInstance(result, MosaicResult)

    def test_overlapping_images_are_averaged(self):
        """Full overlap should average colours instead of overwriting."""
        fp = _make_footprint()
        color1 = (200, 0, 0)
        color2 = (0, 0, 200)
        img1 = _image_for_footprint(fp, color=color1)
        img2 = _image_for_footprint(fp, color=color2)
        stitcher = ImageStitcher([img1, img2], [fp, fp], _CAMERA)
        result = stitcher.stitch()
        centre = result.image[result.image.shape[0] // 2, result.image.shape[1] // 2]
        self.assertGreater(centre[0], 0)
        self.assertGreater(centre[2], 0)
        self.assertLess(centre[0], color1[0])
        self.assertLess(centre[2], color2[2])


# ---------------------------------------------------------------------------
# Test: stitch — error cases
# ---------------------------------------------------------------------------

class TestStitchErrors(unittest.TestCase):

    def test_empty_images_raises(self):
        stitcher = ImageStitcher([], [], _CAMERA)
        with self.assertRaises(ValueError):
            stitcher.stitch()

    def test_mismatched_counts_raises(self):
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        stitcher = ImageStitcher([img, img], [fp], _CAMERA)
        with self.assertRaises(ValueError):
            stitcher.stitch()


# ---------------------------------------------------------------------------
# Test: stitch — refine flag
# ---------------------------------------------------------------------------

class TestStitchRefinement(unittest.TestCase):

    def _make_adjacent_footprints(self):
        fp1 = _make_footprint(lat=_LAT, lon=_LON, alt=_ALT)
        METRES_PER_LAT_DEG = 111_320.0
        avg_lat_rad = math.radians(_LAT)
        metres_per_lon = METRES_PER_LAT_DEG * math.cos(avg_lat_rad)
        lon_step_deg = fp1.ground_width_m / metres_per_lon
        fp2 = _make_footprint(lat=_LAT, lon=_LON + lon_step_deg, alt=_ALT)
        return fp1, fp2

    def test_refine_default_is_false(self):
        """ImageStitcher.refine must default to False."""
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        stitcher = ImageStitcher([img], [fp], _CAMERA)
        self.assertFalse(stitcher.refine)

    def test_refine_no_crash_without_cv2(self):
        """refine=True with cv2 unavailable falls back to GPS-only stitch."""
        import sys
        from unittest.mock import patch
        fp1, fp2 = self._make_adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        stitcher = ImageStitcher([img1, img2], [fp1, fp2], _CAMERA, refine=True)
        with patch.dict(sys.modules, {"cv2": None}):
            result = stitcher.stitch()
        self.assertIsInstance(result, MosaicResult)

    def test_refine_tiny_images_falls_back_to_gps(self):
        """refine=True on tiny images should short-circuit to GPS-only placement."""
        fp1, fp2 = self._make_adjacent_footprints()
        img1 = np.full((120, 180, 3), (255, 0, 0), dtype=np.uint8)
        img2 = np.full((120, 180, 3), (0, 255, 0), dtype=np.uint8)
        stitcher = ImageStitcher([img1, img2], [fp1, fp2], _CAMERA, refine=True)
        result = stitcher.stitch()
        self.assertIsInstance(result, MosaicResult)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
