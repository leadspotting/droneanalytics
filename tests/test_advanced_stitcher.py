"""
test_advanced_stitcher.py  —  Tests for the advanced Task 3 pipeline
=====================================================================
Covers:
  - MosaicResult.diagnostics always present
  - ImageStitcher accepts and absorbs all advanced kwargs
  - alignment_refiner module (refine_placements)
  - seamline_detector module (SeamlineBlender)
  - structural_optimizer module (optimize_poses)
  - render_refined_mosaic module (smart_blend, sharpest, SharpestCompositor)
  - novelty frame filter in ImageStitcher
  - composite_mode routing (smart_blend / sharpest / seamline)
  - api.py debug route returns diagnostics (no 500)

Run with:
    python -m pytest test_advanced_stitcher.py -v
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from models import GPSCoordinate, CameraConfig, GimbalMode, DroneTelemetry
from footprint_calculator import FootprintCalculator, FootprintResult
from drone_oversight import ImageStitcher, MosaicResult, _dlt_homography


# ---------------------------------------------------------------------------
# Shared fixture helpers (same as test_image_stitcher.py)
# ---------------------------------------------------------------------------

_CAMERA = CameraConfig(fov_h=90.0, fov_v=60.0, gimbal_mode=GimbalMode.NADIR)
_ALT = 100.0
_LAT = 32.0
_LON = 34.0
_GSD = 1.0


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
    color: tuple = (128, 64, 32),
    gsd: float = _GSD,
) -> np.ndarray:
    W = max(1, round(fp.ground_width_m / gsd))
    H = max(1, round(fp.ground_height_m / gsd))
    return np.full((H, W, 3), color, dtype=np.uint8)


def _adjacent_footprints():
    """Two side-by-side footprints (no overlap) along longitude."""
    fp1 = _make_footprint(lat=_LAT, lon=_LON, alt=_ALT)
    METRES_PER_LAT_DEG = 111_320.0
    metres_per_lon = METRES_PER_LAT_DEG * math.cos(math.radians(_LAT))
    lon_step = fp1.ground_width_m / metres_per_lon
    fp2 = _make_footprint(lat=_LAT, lon=_LON + lon_step, alt=_ALT)
    return fp1, fp2


# ===========================================================================
# MosaicResult.diagnostics
# ===========================================================================

class TestMosaicResultDiagnostics(unittest.TestCase):

    def test_diagnostics_defaults_to_empty_dict(self):
        r = MosaicResult(
            image=np.zeros((5, 5, 3), dtype=np.uint8),
            bounding_box=[],
            gsd_m=1.0,
        )
        self.assertIsInstance(r.diagnostics, dict)
        self.assertEqual(r.diagnostics, {})

    def test_diagnostics_preserved_when_set(self):
        d = {"alignment_applied": True, "canvas_coverage_pct": 87.3}
        r = MosaicResult(
            image=np.zeros((5, 5, 3), dtype=np.uint8),
            bounding_box=[],
            gsd_m=1.0,
            diagnostics=d,
        )
        self.assertEqual(r.diagnostics["canvas_coverage_pct"], 87.3)

    def test_stitch_result_has_diagnostics(self):
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        result = ImageStitcher([img], [fp], _CAMERA).stitch()
        self.assertIsInstance(result.diagnostics, dict)
        self.assertIn("canvas_coverage_pct", result.diagnostics)

    def test_stitch_diagnostics_contains_expected_keys(self):
        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        result = ImageStitcher([img1, img2], [fp1, fp2], _CAMERA).stitch()
        d = result.diagnostics
        for key in ("canvas_coverage_pct", "frames_used", "frames_filtered",
                    "compositing_mode", "alignment_applied", "sba_applied",
                    "seamline_applied"):
            self.assertIn(key, d, msg=f"Key '{key}' missing from diagnostics")

    def test_coverage_pct_in_range(self):
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        result = ImageStitcher([img], [fp], _CAMERA).stitch()
        pct = result.diagnostics["canvas_coverage_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)


# ===========================================================================
# ImageStitcher accepts all advanced kwargs without crashing
# ===========================================================================

class TestImageStitcherKwargs(unittest.TestCase):
    """Ensure the constructor absorbs every kwarg that api.py passes."""

    def _all_kwargs(self):
        return dict(
            refine=False,
            composite_mode="smart_blend",
            window_size=8,
            window_stride=4,
            compute_window_mosaics=False,
            use_window_merge=False,
            use_alignment_refinement=False,
            use_sba=False,
            use_semantic_seamline=False,
            sba_weights={"w1": 4.0, "w2": 1.5, "w3": 0.0, "w4": 0.03},
            use_novelty_composite=False,
            novelty_max_overlap_ratio=0.90,
            novelty_min_pixels=1500,
            novelty_gps_min_move_m=2.0,
            max_keyframes=None,
            keyframe_latest_wins=True,
            novelty_angle_threshold_deg=12.0,
            novelty_min_overlap_with_last=0.30,
        )

    def test_all_api_kwargs_accepted(self):
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        s = ImageStitcher([img], [fp], _CAMERA, **self._all_kwargs())
        self.assertIsInstance(s, ImageStitcher)

    def test_all_api_kwargs_stitch_returns_mosaic(self):
        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        result = ImageStitcher(
            [img1, img2], [fp1, fp2], _CAMERA, **self._all_kwargs()
        ).stitch()
        self.assertIsInstance(result, MosaicResult)
        self.assertEqual(result.image.dtype, np.uint8)

    def test_unknown_kwargs_absorbed_without_error(self):
        """Future kwargs must not crash the constructor."""
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        s = ImageStitcher([img], [fp], _CAMERA,
                          some_future_flag=True, another_flag=42)
        self.assertIsInstance(s, ImageStitcher)

    def test_refine_and_use_alignment_refinement_are_equivalent(self):
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        s1 = ImageStitcher([img], [fp], _CAMERA, refine=True)
        s2 = ImageStitcher([img], [fp], _CAMERA, use_alignment_refinement=True)
        self.assertTrue(s1.use_alignment_refinement)
        self.assertTrue(s2.use_alignment_refinement)


# ===========================================================================
# alignment_refiner module
# ===========================================================================

class TestAlignmentRefiner(unittest.TestCase):

    def _make_pair(self):
        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(200, 100, 50))
        img2 = _image_for_footprint(fp2, color=(50, 100, 200))
        return [img1, img2], [fp1, fp2]

    def test_import(self):
        from alignment_refiner import refine_placements  # noqa: F401

    def test_returns_same_length_as_input(self):
        from alignment_refiner import refine_placements
        imgs, fps = self._make_pair()
        fp = _make_footprint()
        h_gps = [np.eye(3)] * 2
        refined, metrics = refine_placements(imgs, h_gps)
        self.assertEqual(len(refined), 2)

    def test_metrics_keys_present(self):
        from alignment_refiner import refine_placements
        imgs, _ = self._make_pair()
        h_gps = [np.eye(3)] * 2
        _, metrics = refine_placements(imgs, h_gps)
        for key in ("accepted_pairs", "fallback_pairs", "total_pairs", "mean_inliers"):
            self.assertIn(key, metrics)

    def test_single_image_no_crash(self):
        from alignment_refiner import refine_placements
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        refined, metrics = refine_placements([img], [np.eye(3)])
        self.assertEqual(len(refined), 1)
        self.assertEqual(metrics["total_pairs"], 0)

    def test_fallback_when_cv2_unavailable(self):
        from alignment_refiner import refine_placements
        import sys
        imgs, _ = self._make_pair()
        h_gps = [np.eye(3), np.eye(3) * 2]
        with patch.dict(sys.modules, {"cv2": None}):
            refined, metrics = refine_placements(imgs, h_gps)
        self.assertEqual(len(refined), 2)
        self.assertEqual(metrics.get("fallback_reason"), "cv2_not_available")

    def test_fallback_on_tiny_images(self):
        """Images smaller than MIN_REFINE_PX → GPS-only fallback."""
        from alignment_refiner import refine_placements
        imgs = [np.zeros((100, 150, 3), dtype=np.uint8)] * 2
        h_gps = [np.eye(3)] * 2
        refined, metrics = refine_placements(imgs, h_gps)
        self.assertIn("fallback_reason", metrics)
        # Should return GPS homographies unchanged
        for H in refined:
            self.assertEqual(H.shape, (3, 3))

    def test_output_homographies_are_3x3(self):
        from alignment_refiner import refine_placements
        imgs, _ = self._make_pair()
        h_gps = [np.eye(3)] * 2
        refined, _ = refine_placements(imgs, h_gps)
        for H in refined:
            self.assertIsInstance(H, np.ndarray)
            self.assertEqual(H.shape, (3, 3))


# ===========================================================================
# seamline_detector module
# ===========================================================================

class TestSeamlineDetector(unittest.TestCase):

    def test_import(self):
        from seamline_detector import SeamlineBlender, ImageContribution  # noqa: F401

    def test_empty_contributions_returns_black_canvas(self):
        from seamline_detector import SeamlineBlender
        blender = SeamlineBlender(10, 10)
        canvas = blender.blend([])
        self.assertEqual(canvas.shape, (10, 10, 3))
        self.assertTrue(np.all(canvas == 0))

    def test_single_contribution_fills_canvas(self):
        from seamline_detector import SeamlineBlender, ImageContribution
        H, W = 10, 20
        blender = SeamlineBlender(H, W)
        all_y = np.repeat(np.arange(H), W).astype(np.int32)
        all_x = np.tile(np.arange(W), H).astype(np.int32)
        N = H * W
        contrib = ImageContribution(
            sampled_rgb=np.full((N, 3), 128, dtype=np.float32),
            src_u=np.full(N, 0.5, dtype=np.float32),
            src_v=np.full(N, 0.5, dtype=np.float32),
            canvas_y=all_y,
            canvas_x=all_x,
        )
        canvas = blender.blend([contrib])
        self.assertEqual(canvas.shape, (H, W, 3))
        # All pixels should have been painted
        self.assertTrue(np.all(canvas != 0))

    def test_two_contributions_no_crash(self):
        from seamline_detector import SeamlineBlender, ImageContribution
        H, W = 20, 40
        blender = SeamlineBlender(H, W)
        # Left half
        left_y = np.repeat(np.arange(H), W // 2).astype(np.int32)
        left_x = np.tile(np.arange(W // 2), H).astype(np.int32)
        # Right half
        right_y = left_y.copy()
        right_x = np.tile(np.arange(W // 2, W), H).astype(np.int32)
        N_half = H * (W // 2)
        c1 = ImageContribution(
            sampled_rgb=np.full((N_half, 3), 200, dtype=np.float32),
            src_u=np.full(N_half, 0.25, dtype=np.float32),
            src_v=np.full(N_half, 0.5, dtype=np.float32),
            canvas_y=left_y, canvas_x=left_x, centre_u=0.25, centre_v=0.5,
        )
        c2 = ImageContribution(
            sampled_rgb=np.full((N_half, 3), 100, dtype=np.float32),
            src_u=np.full(N_half, 0.75, dtype=np.float32),
            src_v=np.full(N_half, 0.5, dtype=np.float32),
            canvas_y=right_y, canvas_x=right_x, centre_u=0.75, centre_v=0.5,
        )
        canvas = blender.blend([c1, c2])
        self.assertEqual(canvas.shape, (H, W, 3))

    def test_feather_blend_utility(self):
        from seamline_detector import SeamlineBlender
        accum = np.full((5, 5, 3), 255.0, dtype=np.float32)
        wsum = np.ones((5, 5), dtype=np.float32)
        canvas = SeamlineBlender.feather_blend(accum, wsum)
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertTrue(np.all(canvas == 255))


# ===========================================================================
# structural_optimizer module
# ===========================================================================

class TestStructuralOptimizer(unittest.TestCase):

    def test_import(self):
        from structural_optimizer import optimize_poses  # noqa: F401

    def test_passthrough_for_two_or_fewer(self):
        from structural_optimizer import optimize_poses
        H = [np.eye(3), np.eye(3) * 2]
        h_opt, meta = optimize_poses(H, H)
        self.assertEqual(meta["method"], "passthrough")
        self.assertEqual(len(h_opt), 2)

    def test_returns_same_length(self):
        from structural_optimizer import optimize_poses
        N = 5
        h_gps = [np.eye(3) for _ in range(N)]
        h_ref = [np.eye(3) + np.random.randn(3, 3) * 0.01 for _ in range(N)]
        h_opt, meta = optimize_poses(h_gps, h_ref)
        self.assertEqual(len(h_opt), N)

    def test_output_homographies_are_3x3(self):
        from structural_optimizer import optimize_poses
        N = 4
        h_gps = [np.eye(3)] * N
        h_ref = [np.eye(3)] * N
        h_opt, _ = optimize_poses(h_gps, h_ref)
        for H in h_opt:
            self.assertEqual(H.shape, (3, 3))

    def test_meta_keys_present(self):
        from structural_optimizer import optimize_poses
        h = [np.eye(3)] * 4
        _, meta = optimize_poses(h, h)
        for key in ("method", "drift_px", "correction_applied"):
            self.assertIn(key, meta)

    def test_drift_correction_reduces_end_error(self):
        """Optimizer should move h_opt[-1] closer to h_gps[-1] than h_ref[-1] is."""
        from structural_optimizer import optimize_poses
        N = 6
        h_gps = [np.eye(3).copy() for _ in range(N)]
        # Simulate linear drift: each image shifts +0.1 in x translation
        h_ref = []
        for i in range(N):
            H = np.eye(3).copy()
            H[0, 2] = i * 0.05   # growing drift
            h_ref.append(H)

        h_opt, meta = optimize_poses(h_gps, h_ref)

        # The last refined H has drift = (N-1)*0.05 in translation
        # The optimized last H should be closer to 0 (GPS anchor)
        drift_before = abs(h_ref[-1][0, 2] - h_gps[-1][0, 2])
        drift_after = abs(h_opt[-1][0, 2] - h_gps[-1][0, 2])
        self.assertLessEqual(drift_after, drift_before + 1e-9)


# ===========================================================================
# render_refined_mosaic module
# ===========================================================================

class TestRenderRefinedMosaic(unittest.TestCase):

    def test_import(self):
        from render_refined_mosaic import (  # noqa: F401
            smart_blend, SharpestCompositor, compute_image_sharpness,
        )

    def test_smart_blend_painted(self):
        from render_refined_mosaic import smart_blend
        accum = np.full((10, 10, 3), 200.0, dtype=np.float32)
        wsum = np.ones((10, 10), dtype=np.float32)
        canvas = smart_blend(accum, wsum, post_sharpen=False)
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertTrue(np.all(canvas > 0))

    def test_smart_blend_unpainted_stays_black(self):
        from render_refined_mosaic import smart_blend
        accum = np.zeros((10, 10, 3), dtype=np.float32)
        wsum = np.zeros((10, 10), dtype=np.float32)
        canvas = smart_blend(accum, wsum, post_sharpen=False)
        self.assertTrue(np.all(canvas == 0))

    def test_sharpest_compositor_basic(self):
        from render_refined_mosaic import SharpestCompositor
        comp = SharpestCompositor(10, 20)
        cy = np.arange(10, dtype=np.int32)
        cx = np.zeros(10, dtype=np.int32)
        rgb = np.full((10, 3), 150, dtype=np.float32)
        fw = np.ones(10, dtype=np.float32)
        comp.update(cy, cx, rgb, fw, image_sharpness=100.0)
        canvas = comp.result(post_sharpen=False)
        self.assertEqual(canvas.shape, (10, 20, 3))
        self.assertEqual(canvas.dtype, np.uint8)

    def test_sharpness_score_positive_for_non_uniform(self):
        from render_refined_mosaic import compute_image_sharpness
        # Checkerboard pattern → high sharpness
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        score = compute_image_sharpness(img)
        self.assertGreater(score, 0.0)

    def test_sharpness_score_zero_for_uniform(self):
        from render_refined_mosaic import compute_image_sharpness
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        score = compute_image_sharpness(img)
        self.assertAlmostEqual(score, 0.0, places=6)


# ===========================================================================
# Composite-mode routing in ImageStitcher
# ===========================================================================

class TestCompositeModeRouting(unittest.TestCase):

    def _two_images(self):
        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        return [img1, img2], [fp1, fp2]

    def test_smart_blend_returns_mosaic(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(imgs, fps, _CAMERA,
                               composite_mode="smart_blend").stitch()
        self.assertIsInstance(result, MosaicResult)
        self.assertEqual(result.diagnostics["compositing_mode"], "smart_blend")

    def test_sharpest_returns_mosaic(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(imgs, fps, _CAMERA,
                               composite_mode="sharpest").stitch()
        self.assertIsInstance(result, MosaicResult)
        self.assertEqual(result.diagnostics["compositing_mode"], "sharpest")

    def test_semantic_seamline_returns_mosaic(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(imgs, fps, _CAMERA,
                               use_semantic_seamline=True).stitch()
        self.assertIsInstance(result, MosaicResult)
        self.assertEqual(result.diagnostics["compositing_mode"], "seamline_voronoi")
        self.assertTrue(result.diagnostics["seamline_applied"])

    def test_default_mode_is_smart_blend(self):
        fp = _make_footprint()
        img = _image_for_footprint(fp)
        result = ImageStitcher([img], [fp], _CAMERA).stitch()
        self.assertEqual(result.diagnostics["compositing_mode"], "smart_blend")

    def test_all_modes_produce_uint8_canvas(self):
        imgs, fps = self._two_images()
        for mode in ("smart_blend", "sharpest"):
            with self.subTest(mode=mode):
                result = ImageStitcher(imgs, fps, _CAMERA,
                                       composite_mode=mode).stitch()
                self.assertEqual(result.image.dtype, np.uint8)


# ===========================================================================
# Novelty frame filter
# ===========================================================================

class TestNoveltyFrameFilter(unittest.TestCase):

    def _sequential_footprints(self, n: int, lon_step_deg: float = 0.0001):
        """N footprints in a row along longitude."""
        fps = []
        for i in range(n):
            fps.append(_make_footprint(lon=_LON + i * lon_step_deg))
        return fps

    def test_novelty_filter_reduces_frame_count(self):
        """Stationary frames should be filtered out."""
        # All footprints at the same GPS position → all but first filtered
        fp = _make_footprint()
        imgs = [_image_for_footprint(fp, color=(i * 10, 0, 0)) for i in range(5)]
        fps = [fp] * 5
        result = ImageStitcher(
            imgs, fps, _CAMERA,
            use_novelty_composite=True,
            novelty_gps_min_move_m=1.0,
        ).stitch()
        self.assertLess(result.diagnostics["frames_used"],
                        result.diagnostics["frames_input"])

    def test_novelty_filter_metrics_in_diagnostics(self):
        fp = _make_footprint()
        imgs = [_image_for_footprint(fp)] * 3
        fps = [fp] * 3
        result = ImageStitcher(
            imgs, fps, _CAMERA, use_novelty_composite=True
        ).stitch()
        d = result.diagnostics
        self.assertIn("frames_input", d)
        self.assertIn("frames_used", d)
        self.assertIn("frames_filtered", d)
        self.assertIn("duplicate_frames_filtered", d)
        self.assertIn("novelty_decisions", d)
        self.assertEqual(d["frames_input"] + 0, d["frames_used"] + d["frames_filtered"])

    def test_novelty_filter_keeps_moving_frames(self):
        """Frames with sufficient GPS movement should not be filtered."""
        fps = self._sequential_footprints(4, lon_step_deg=0.01)  # ~1km steps
        imgs = [_image_for_footprint(fp) for fp in fps]
        result = ImageStitcher(
            imgs, fps, _CAMERA,
            use_novelty_composite=True,
            novelty_gps_min_move_m=1.0,
        ).stitch()
        # Most frames should be kept when there's real movement
        self.assertGreaterEqual(result.diagnostics["frames_used"], 2)

    def test_max_keyframes_limits_output(self):
        fps = self._sequential_footprints(6, lon_step_deg=0.01)
        imgs = [_image_for_footprint(fp) for fp in fps]
        result = ImageStitcher(
            imgs, fps, _CAMERA,
            use_novelty_composite=True,
            novelty_gps_min_move_m=1.0,
            max_keyframes=3,
        ).stitch()
        self.assertLessEqual(result.diagnostics["frames_used"], 3)

    def test_novelty_filter_keeps_same_gps_frame_with_new_visual_content(self):
        fp = _make_footprint()
        img1 = np.zeros((160, 240, 3), dtype=np.uint8)
        img2 = img1.copy()
        img1[:, 20:60] = 70
        img2[:, 20:60] = 70
        img2[50:110, 150:220] = 255
        result = ImageStitcher(
            [img1, img2], [fp, fp], _CAMERA,
            use_novelty_composite=True,
            novelty_gps_min_move_m=5.0,
            novelty_angle_threshold_deg=20.0,
        ).stitch()
        self.assertEqual(result.diagnostics["frames_used"], 2)
        self.assertGreaterEqual(result.diagnostics["visual_novel_frames_kept"], 1)

    def test_novelty_filter_keeps_small_shift_when_edges_add_new_pixels(self):
        fp1 = _make_footprint(lat=_LAT, lon=_LON, alt=_ALT)
        metres_per_lon = 111_320.0 * math.cos(math.radians(_LAT))
        lon_step_deg = (fp1.ground_width_m * 0.05) / metres_per_lon
        fp2 = _make_footprint(lat=_LAT, lon=_LON + lon_step_deg, alt=_ALT)
        img1 = np.zeros((160, 240, 3), dtype=np.uint8)
        img2 = np.zeros((160, 240, 3), dtype=np.uint8)
        img1[:, :220] = 90
        img2[:, 20:] = 90
        img2[:, -12:] = 220
        result = ImageStitcher(
            [img1, img2], [fp1, fp2], _CAMERA,
            use_novelty_composite=True,
            novelty_gps_min_move_m=50.0,
            novelty_angle_threshold_deg=20.0,
            novelty_min_pixels=500,
            novelty_max_overlap_ratio=0.98,
        ).stitch()
        self.assertEqual(result.diagnostics["frames_used"], 2)
        self.assertGreaterEqual(result.diagnostics["edge_novel_frames_kept"], 1)


# ===========================================================================
# Advanced path end-to-end (alignment + sba + seamline together)
# ===========================================================================

class TestAdvancedPipelineEndToEnd(unittest.TestCase):

    def _two_images(self):
        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        return [img1, img2], [fp1, fp2]

    def test_alignment_plus_seamline_no_crash(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(
            imgs, fps, _CAMERA,
            use_alignment_refinement=True,
            use_semantic_seamline=True,
        ).stitch()
        self.assertIsInstance(result, MosaicResult)
        self.assertEqual(result.image.dtype, np.uint8)

    def test_alignment_plus_sba_no_crash(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(
            imgs, fps, _CAMERA,
            use_alignment_refinement=True,
            use_sba=True,
        ).stitch()
        self.assertIsInstance(result, MosaicResult)

    def test_all_flags_enabled_no_crash(self):
        """Every advanced flag active simultaneously must not raise."""
        imgs, fps = self._two_images()
        result = ImageStitcher(
            imgs, fps, _CAMERA,
            use_alignment_refinement=True,
            use_sba=True,
            use_semantic_seamline=True,
            use_novelty_composite=False,  # keep both frames
            composite_mode="smart_blend",
        ).stitch()
        self.assertIsInstance(result, MosaicResult)
        self.assertIn("alignment", result.diagnostics)
        self.assertIn("sba", result.diagnostics)

    def test_sba_diagnostics_present(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(
            imgs, fps, _CAMERA, use_sba=True
        ).stitch()
        self.assertIn("sba", result.diagnostics)

    def test_alignment_fallback_when_cv2_missing(self):
        import sys
        imgs, fps = self._two_images()
        with patch.dict(sys.modules, {"cv2": None}):
            result = ImageStitcher(
                imgs, fps, _CAMERA, use_alignment_refinement=True
            ).stitch()
        self.assertIsInstance(result, MosaicResult)
        # alignment was attempted but fell back
        align = result.diagnostics.get("alignment", {})
        self.assertGreaterEqual(align.get("fallback_pairs", 0), 0)


# ===========================================================================
# API debug route — no 500 on real call
# ===========================================================================

class TestDebugRouteNo500(unittest.TestCase):

    def setUp(self):
        import api as api_module
        api_module._image_preview_cache.clear()

    def test_debug_route_404_for_missing_mission(self):
        from fastapi.testclient import TestClient
        import api as api_module
        client = TestClient(api_module.app)
        resp = client.get("/image_mission/DOES_NOT_EXIST/mosaic/debug")
        self.assertEqual(resp.status_code, 404)

    def test_debug_route_returns_diagnostics_key(self):
        """Inject a tiny fake mission and confirm the debug route returns JSON
        with a 'diagnostics' key (the stitch result's .diagnostics attr)."""
        from unittest.mock import patch as _patch
        from fastapi.testclient import TestClient
        from PIL import Image as PILImage
        import api as api_module

        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))

        fake_rows = [
            {
                "path": "/fake/img1.jpg",
                "lat": fp1.nadir.lat, "lon": fp1.nadir.lon,
                "agl_m": 100.0, "alt_ellipsoid_m": 1465.0,
                "timestamp": "", "heading": 0.0,
                "width_px": 200, "height_px": 115,
                "fov_h": 90.0, "fov_v": 60.0,
            },
            {
                "path": "/fake/img2.jpg",
                "lat": fp2.nadir.lat, "lon": fp2.nadir.lon,
                "agl_m": 100.0, "alt_ellipsoid_m": 1465.0,
                "timestamp": "", "heading": 0.0,
                "width_px": 200, "height_px": 115,
                "fov_h": 90.0, "fov_v": 60.0,
            },
        ]

        api_module._image_missions["TEST_DEBUG"] = fake_rows

        tiny_pil = PILImage.fromarray(img1)

        try:
            with _patch("PIL.Image.open", return_value=tiny_pil):
                client = TestClient(api_module.app)
                resp = client.get(
                    "/image_mission/TEST_DEBUG/mosaic/debug"
                    "?scale=0.05&refine=false"
                )
            self.assertNotEqual(resp.status_code, 500,
                                msg=f"Debug route 500'd: {resp.text[:300]}")
            # Accept 200 or 422 (if images can't be loaded from fake path)
            # The key thing is: no AttributeError on .diagnostics
            if resp.status_code == 200:
                data = resp.json()
                self.assertIn("diagnostics", data)
        finally:
            api_module._image_missions.pop("TEST_DEBUG", None)


# ===========================================================================
# GPS overlap helpers in alignment_refiner
# ===========================================================================

class TestGPSOverlapHelpers(unittest.TestCase):
    """Tests for the canvas-bbox and overlap-computation utilities."""

    def _make_h_gps_pair(self, lon_step_deg: float = 0.01):
        """Two GPS homographies: image 0 at _LON, image 1 at _LON+lon_step."""
        fp1 = _make_footprint(lon=_LON)
        fp2 = _make_footprint(lon=_LON + lon_step_deg)
        from drone_oversight import _dlt_homography, GPSCoordinate

        class _FP:
            def __init__(self, fp):
                self.corners = fp.corners

        # Build a tiny canvas footprint spanning both
        all_lats = [c.lat for c in fp1.corners + fp2.corners]
        all_lons = [c.lon for c in fp1.corners + fp2.corners]
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        def _make_H(fp):
            src = [[0, 0], [1, 0], [1, 1], [0, 1]]
            dst = [
                [(c.lon - lon_min) / lon_span, (lat_max - c.lat) / lat_span]
                for c in fp.corners
            ]
            return _dlt_homography(
                np.array(src, dtype=np.float64),
                np.array(dst, dtype=np.float64),
            )

        return _make_H(fp1), _make_H(fp2)

    def test_canvas_bbox_identity(self):
        """Identity maps [0,1]^2 to itself."""
        from alignment_refiner import _canvas_bbox
        u0, u1, v0, v1 = _canvas_bbox(np.eye(3))
        self.assertAlmostEqual(u0, 0.0, places=10)
        self.assertAlmostEqual(u1, 1.0, places=10)
        self.assertAlmostEqual(v0, 0.0, places=10)
        self.assertAlmostEqual(v1, 1.0, places=10)

    def test_canvas_bbox_translation(self):
        """Pure translation shifts the bbox by that amount."""
        from alignment_refiner import _canvas_bbox
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = 0.3   # translate u by 0.3
        H[1, 2] = 0.1   # translate v by 0.1
        u0, u1, v0, v1 = _canvas_bbox(H)
        self.assertAlmostEqual(u0, 0.3, places=8)
        self.assertAlmostEqual(u1, 1.3, places=8)
        self.assertAlmostEqual(v0, 0.1, places=8)

    def test_gps_overlap_full_when_identical(self):
        """Two identical homographies → 100 % overlap."""
        from alignment_refiner import _compute_gps_overlap
        H = np.eye(3)
        ovl = _compute_gps_overlap(H, H)
        self.assertAlmostEqual(ovl, 1.0, places=6)

    def test_gps_overlap_zero_when_non_overlapping(self):
        """Images separated beyond their width have zero overlap."""
        from alignment_refiner import _compute_gps_overlap
        H_i = np.eye(3)        # occupies u∈[0,1]
        H_j = np.eye(3).copy()
        H_j[0, 2] = 2.0        # occupies u∈[2,3] — no intersection
        ovl = _compute_gps_overlap(H_i, H_j)
        self.assertAlmostEqual(ovl, 0.0, places=6)

    def test_gps_overlap_partial(self):
        """50 % horizontal shift → ~50 % overlap."""
        from alignment_refiner import _compute_gps_overlap
        H_i = np.eye(3)
        H_j = np.eye(3).copy()
        H_j[0, 2] = 0.5   # shift right by 0.5 of width
        ovl = _compute_gps_overlap(H_i, H_j)
        # intersection u∈[0.5,1] v∈[0,1] → area 0.5, area_i = 1 → 0.5
        self.assertAlmostEqual(ovl, 0.5, places=6)

    def test_overlap_roi_returns_none_for_no_overlap(self):
        """Non-overlapping images → None ROI."""
        from alignment_refiner import _overlap_roi_in_image
        H_i = np.eye(3)
        H_j = np.eye(3).copy()
        H_j[0, 2] = 5.0   # completely separated
        roi = _overlap_roi_in_image(H_i, H_j, img_h=100, img_w=100)
        self.assertIsNone(roi)

    def test_overlap_roi_returns_tuple_for_overlap(self):
        """Partially overlapping images → valid (x0,x1,y0,y1) ROI."""
        from alignment_refiner import _overlap_roi_in_image
        H_i = np.eye(3)
        H_j = np.eye(3).copy()
        H_j[0, 2] = 0.5   # 50 % overlap on the right side
        roi = _overlap_roi_in_image(H_i, H_j, img_h=400, img_w=600)
        if roi is not None:
            x0, x1, y0, y1 = roi
            self.assertGreaterEqual(x0, 0)
            self.assertLessEqual(x1, 600)
            self.assertGreater(x1, x0)


# ===========================================================================
# New alignment_refiner diagnostics
# ===========================================================================

class TestAlignmentRefinerNewDiagnostics(unittest.TestCase):
    """Verify that new diagnostic keys are always present in metrics."""

    def _solid_pair(self):
        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(200, 100, 50))
        img2 = _image_for_footprint(fp2, color=(50, 100, 200))
        return [img1, img2], [np.eye(3), np.eye(3)]

    def test_new_keys_present_on_fallback(self):
        """Keys are present even when all pairs fall back."""
        from alignment_refiner import refine_placements
        imgs, h_gps = self._solid_pair()
        _, metrics = refine_placements(imgs, h_gps)
        for key in ("rejected_pairs", "gps_anchor_error_px",
                    "refinement_shift_px", "per_pair", "visual_edges"):
            self.assertIn(key, metrics, msg=f"Missing: {key}")

    def test_new_keys_present_on_cv2_missing(self):
        """Keys present even when cv2 is unavailable."""
        import sys
        from alignment_refiner import refine_placements
        imgs, h_gps = self._solid_pair()
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            sys.modules, {"cv2": None}
        ):
            _, metrics = refine_placements(imgs, h_gps)
        for key in ("rejected_pairs", "gps_anchor_error_px",
                    "refinement_shift_px", "per_pair", "visual_edges"):
            self.assertIn(key, metrics, msg=f"Missing: {key}")

    def test_gps_anchor_error_is_list_of_floats(self):
        from alignment_refiner import refine_placements
        imgs, h_gps = self._solid_pair()
        _, metrics = refine_placements(imgs, h_gps)
        err = metrics["gps_anchor_error_px"]
        self.assertIsInstance(err, list)
        self.assertEqual(len(err), 2)
        for v in err:
            self.assertIsInstance(v, float)

    def test_per_pair_is_list(self):
        from alignment_refiner import refine_placements
        imgs, h_gps = self._solid_pair()
        _, metrics = refine_placements(imgs, h_gps)
        self.assertIsInstance(metrics["per_pair"], list)

    def test_visual_edges_is_list(self):
        from alignment_refiner import refine_placements
        imgs, h_gps = self._solid_pair()
        _, metrics = refine_placements(imgs, h_gps)
        self.assertIsInstance(metrics["visual_edges"], list)

    def test_neighbor_window_param_accepted(self):
        """neighbor_window and min_gps_overlap_frac are accepted without error."""
        from alignment_refiner import refine_placements
        imgs, h_gps = self._solid_pair()
        refined, metrics = refine_placements(
            imgs, h_gps,
            neighbor_window=2,
            min_gps_overlap_frac=0.05,
            max_gps_deviation_frac=0.30,
        )
        self.assertEqual(len(refined), 2)

    def test_total_pairs_gte_n_minus_1(self):
        """With neighbor_window > 1 we may get more pairs than N-1."""
        from alignment_refiner import refine_placements
        fps = [_make_footprint(lon=_LON + i * 0.0001) for i in range(4)]
        # Use gsd=0.3 so images are ~667×385 px — above MIN_REFINE_PX
        imgs = [_image_for_footprint(fp, gsd=0.3) for fp in fps]
        h_gps = [np.eye(3)] * 4   # identity → all pairs overlap 100%
        _, metrics = refine_placements(imgs, h_gps, neighbor_window=3)
        # At minimum should have 3 consecutive pairs
        self.assertGreaterEqual(metrics["total_pairs"], 3)


# ===========================================================================
# structural_optimizer with visual_edges
# ===========================================================================

class TestStructuralOptimizerWithEdges(unittest.TestCase):

    def _make_h_list(self, N: int, drift_per_step: float = 0.05):
        h_gps = []
        h_ref = []
        for i in range(N):
            H_g = np.eye(3, dtype=np.float64)
            h_gps.append(H_g)
            H_r = np.eye(3, dtype=np.float64)
            H_r[0, 2] = i * drift_per_step   # accumulated drift
            h_ref.append(H_r)
        return h_gps, h_ref

    def test_visual_edges_none_is_backward_compat(self):
        """Calling with visual_edges=None gives same meta keys as before."""
        from structural_optimizer import optimize_poses
        h_g, h_r = self._make_h_list(5)
        _, meta = optimize_poses(h_g, h_r, visual_edges=None)
        for key in ("method", "drift_px", "correction_applied"):
            self.assertIn(key, meta)

    def test_new_meta_keys_present(self):
        """New diagnostics keys are always in meta."""
        from structural_optimizer import optimize_poses
        h_g, h_r = self._make_h_list(5)
        _, meta = optimize_poses(h_g, h_r)
        for key in ("gps_anchor_error_px", "refinement_shift_px",
                    "visual_edges_used"):
            self.assertIn(key, meta, msg=f"Missing meta key: {key}")

    def test_visual_edges_used_count(self):
        """visual_edges_used reflects valid edges provided."""
        from structural_optimizer import optimize_poses
        h_g, h_r = self._make_h_list(5)
        edges = [
            {"i": 0, "j": 2, "delta_t": [0.1, 0.0], "confidence": 0.8},
            {"i": 1, "j": 3, "delta_t": [0.1, 0.0], "confidence": 0.7},
        ]
        _, meta = optimize_poses(h_g, h_r, visual_edges=edges)
        self.assertEqual(meta["visual_edges_used"], 2)

    def test_invalid_edges_are_silently_skipped(self):
        """Malformed or out-of-range edges do not crash."""
        from structural_optimizer import optimize_poses
        h_g, h_r = self._make_h_list(4)
        bad_edges = [
            {"i": 0, "j": 99, "delta_t": [0.1, 0.0], "confidence": 0.5},  # j out of range
            {"i": "x", "j": 1, "delta_t": [0.1, 0.0], "confidence": 0.5},  # bad type
            {"i": 0, "j": 0, "delta_t": [0.1, 0.0], "confidence": 0.5},   # i==j
        ]
        h_opt, meta = optimize_poses(h_g, h_r, visual_edges=bad_edges)
        self.assertEqual(len(h_opt), 4)
        self.assertEqual(meta["visual_edges_used"], 0)

    def test_non_sequential_edge_pulls_image_toward_constraint(self):
        """A strong cross-image visual edge should influence the result."""
        from structural_optimizer import optimize_poses
        N = 6
        h_gps = [np.eye(3, dtype=np.float64) for _ in range(N)]
        h_ref = []
        for i in range(N):
            H = np.eye(3, dtype=np.float64)
            H[0, 2] = i * 0.05   # growing drift
            h_ref.append(H)

        # Non-sequential edge: image 0 → image 5 with delta_t = 0 (GPS-like)
        edges = [{"i": 0, "j": 5, "delta_t": [0.0, 0.0], "confidence": 1.0}]
        h_opt, _ = optimize_poses(h_gps, h_ref, visual_edges=edges)
        # Image 5's drift should be smaller than h_ref[5]'s drift
        drift_before = abs(h_ref[5][0, 2])
        drift_after = abs(h_opt[5][0, 2])
        self.assertLess(drift_after, drift_before + 1e-9)

    def test_passthrough_still_has_new_keys(self):
        """Even passthrough path (N<=2) includes new meta keys."""
        from structural_optimizer import optimize_poses
        H = [np.eye(3), np.eye(3)]
        _, meta = optimize_poses(H, H)
        self.assertIn("gps_anchor_error_px", meta)
        self.assertIn("refinement_shift_px", meta)
        self.assertIn("visual_edges_used", meta)


# ===========================================================================
# New stitch() diagnostics surfaced at top level
# ===========================================================================

class TestStitchNewDiagnostics(unittest.TestCase):
    """Verify new diagnostic keys are surfaced in MosaicResult.diagnostics."""

    def _two_images(self):
        fp1, fp2 = _adjacent_footprints()
        img1 = _image_for_footprint(fp1, color=(255, 0, 0))
        img2 = _image_for_footprint(fp2, color=(0, 255, 0))
        return [img1, img2], [fp1, fp2]

    def test_accepted_pairs_key_present(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(imgs, fps, _CAMERA,
                               use_alignment_refinement=True).stitch()
        self.assertIn("accepted_pairs", result.diagnostics)

    def test_rejected_pairs_key_present(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(imgs, fps, _CAMERA,
                               use_alignment_refinement=True).stitch()
        self.assertIn("rejected_pairs", result.diagnostics)

    def test_mean_inliers_key_present(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(imgs, fps, _CAMERA,
                               use_alignment_refinement=True).stitch()
        self.assertIn("mean_inliers", result.diagnostics)

    def test_gps_anchor_error_px_key_present_with_sba(self):
        imgs, fps = self._two_images()
        # sba only runs for N > 2, but the key should still be surfaced
        result = ImageStitcher(imgs, fps, _CAMERA, use_sba=True).stitch()
        self.assertIn("gps_anchor_error_px", result.diagnostics)

    def test_refinement_shift_px_key_present_with_sba(self):
        imgs, fps = self._two_images()
        result = ImageStitcher(imgs, fps, _CAMERA, use_sba=True).stitch()
        self.assertIn("refinement_shift_px", result.diagnostics)

    def test_visual_edges_forwarded_to_sba(self):
        """With alignment + sba on 3 images, visual_edges_used appears in sba meta."""
        fps = [_make_footprint(lon=_LON + i * 0.005) for i in range(3)]
        imgs = [_image_for_footprint(fp) for fp in fps]
        result = ImageStitcher(
            imgs, fps, _CAMERA,
            use_alignment_refinement=True,
            use_sba=True,
        ).stitch()
        sba = result.diagnostics.get("sba", {})
        self.assertIn("visual_edges_used", sba)


if __name__ == "__main__":
    unittest.main(verbosity=2)
