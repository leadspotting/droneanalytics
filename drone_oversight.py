"""
drone_oversight.py  —  System Skeleton (Tasks 1–6)
====================================================
Central orchestrator for the Drone Oversight system.

Data-flow diagram
-----------------

  DroneTelemetry
       │
       ▼
  ┌────────────────────┐
  │  FootprintCalculator │  Task 1  ──► FootprintResult (4 GPS corners)
  └────────────────────┘                        │
                                                ▼
                                  ┌─────────────────────────┐
                                  │  CoverageHeatmap         │  Task 2
                                  │  (grid of GPSCoordinate) │
                                  └─────────────────────────┘
                                                │
                     ┌──────────────────────────┘
                     │
                     ▼
          ┌───────────────────┐
          │  ImageStitcher     │  Task 3  ──► MosaicResult (stitched image)
          └───────────────────┘                      │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │  GPSProjector            │  Task 4
                                        │  (pixel → GPS coordinate)│
                                        └─────────────────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │  AltitudeScaler          │  Task 5
                                        │  (adaptive GSD scaling)  │
                                        └─────────────────────────┘

  Area-of-Interest (GPSCoordinate boundary)
       │
       ▼
  ┌────────────────────┐
  │  MissionPlanner     │  Task 6  ──► list[GPSCoordinate] waypoints
  └────────────────────┘
"""

from __future__ import annotations

from typing import Optional

import math

import numpy as np

from models import GPSCoordinate, DroneTelemetry, CameraConfig, GimbalMode
from coverage_heatmap import CoverageHeatmap
from mission_planner import MissionPlanner
import progress

# Upper bound on mosaic canvas size (pixels); larger canvases are coarsened.
MAX_CANVAS_PIXELS = 40_000_000
# Pixels processed per strip when warping an image onto the canvas.
STRIP_PIXELS = 250_000


# ===========================================================================
# Task 2 – Coverage Heatmap  (implemented in coverage_heatmap.py)
# ===========================================================================
# CoverageHeatmap is imported above from coverage_heatmap.py


# ===========================================================================
# Task 3 – Image Stitching & Mosaicking
# ===========================================================================

class MosaicResult:
    """
    Container for the result of image stitching.

    Attributes
    ----------
    image : np.ndarray
        The stitched mosaic image (H×W×3 uint8 or float32).
    bounding_box : list[GPSCoordinate]
        Four GPS corners of the mosaic.
    gsd_m : float
        Ground sampling distance in metres/pixel.
    diagnostics : dict
        Pipeline metrics:  alignment stats, compositing mode, novelty filter
        counts, canvas coverage percentage, etc.  Always present (empty dict
        for the basic GPS-only path).
    """

    def __init__(
        self,
        image: np.ndarray,
        bounding_box: list[GPSCoordinate],
        gsd_m: float,
        diagnostics: Optional[dict] = None,
    ) -> None:
        self.image = image
        self.bounding_box = bounding_box
        self.gsd_m = gsd_m
        self.diagnostics: dict = diagnostics if diagnostics is not None else {}


def _dlt_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Solve a 3×3 homography H from 4 point correspondences via DLT (SVD).

    Parameters
    ----------
    src_pts : np.ndarray  shape (4, 2)  Source points (Euclidean).
    dst_pts : np.ndarray  shape (4, 2)  Destination points (Euclidean).

    Returns
    -------
    np.ndarray  shape (3, 3)  H such that dst ≈ H @ src (homogeneous),
                              normalised so H[2, 2] = 1.
    """
    A = []
    for (xs, ys), (xd, yd) in zip(src_pts, dst_pts):
        A.append([-xs, -ys, -1,   0,   0,  0, xd * xs, xd * ys, xd])
        A.append([  0,   0,  0, -xs, -ys, -1, yd * xs, yd * ys, yd])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


class ImageStitcher:
    """
    Task 3: Stitches individual drone images into a single geo-referenced mosaic.

    Strategy: GPS-guided inverse warp (pure numpy, no OpenCV required for the
    basic path).  Advanced kwargs enable optional refinement stages:

      use_alignment_refinement / refine
          Consecutive pairwise ORB feature matching (requires cv2).
          Falls back to GPS-only when cv2 is absent or images are too small.

      use_sba
          Global pose graph optimization on top of ORB placements.
          Scipy-based least-squares when available; linear drift correction otherwise.

      use_semantic_seamline
          Voronoi seam-cut compositing: assigns each canvas pixel to the nearest
          image centre, feather-blends at seam boundaries.  Pure numpy.

      composite_mode
          "smart_blend" (feather-weighted average, default for advanced path),
          "sharpest"    (per-pixel sharpest-source selection),
          "last_painter_wins" (basic overlay, GPS-only default).

      use_novelty_composite
          Filter input frames before stitching: skip frames that are too close
          to the previous accepted frame (GPS distance + angle threshold).

    All advanced options degrade gracefully to GPS-only when dependencies are
    unavailable or images are below the minimum size for feature matching.

    Inputs
    ------
    images         : list[np.ndarray]      — raw captured images (H×W×3 uint8)
    footprints     : list[FootprintResult] — one per image (from Task 1)
    camera_config  : CameraConfig

    Output
    ------
    MosaicResult  (always includes .diagnostics dict)
    """

    def __init__(
        self,
        images: list[np.ndarray],
        footprints: list[FootprintResult],
        camera_config: CameraConfig,
        # ── Backward-compatible basic option ─────────────────────────────────
        refine: bool = False,
        # ── Advanced pipeline flags ───────────────────────────────────────────
        use_alignment_refinement: bool = False,
        use_sba: bool = False,
        use_semantic_seamline: bool = False,
        composite_mode: str = "smart_blend",
        sba_weights: Optional[dict] = None,
        # ── Novelty frame filter ──────────────────────────────────────────────
        use_novelty_composite: bool = False,
        novelty_gps_min_move_m: float = 2.0,
        novelty_angle_threshold_deg: float = 12.0,
        novelty_max_overlap_ratio: float = 0.90,
        novelty_min_pixels: int = 1500,
        novelty_min_overlap_with_last: float = 0.30,
        max_keyframes: Optional[int] = None,
        keyframe_latest_wins: bool = True,
        # ── Window merge (accepted but not yet implemented; degrades gracefully) ─
        use_window_merge: bool = False,
        compute_window_mosaics: bool = False,
        window_size: int = 8,
        window_stride: int = 4,
        # ── Absorb any unknown future kwargs without crashing ─────────────────
        **_ignored_kwargs,
    ) -> None:
        self.images = images
        self.footprints = footprints
        self.camera_config = camera_config
        # refine and use_alignment_refinement are equivalent entry-points
        self.refine = refine or use_alignment_refinement
        self.use_alignment_refinement = use_alignment_refinement or refine
        self.use_sba = use_sba
        self.use_semantic_seamline = use_semantic_seamline
        self.composite_mode = composite_mode
        self.sba_weights = sba_weights or {}
        self.use_novelty_composite = use_novelty_composite
        self.novelty_gps_min_move_m = novelty_gps_min_move_m
        self.novelty_angle_threshold_deg = novelty_angle_threshold_deg
        self.novelty_max_overlap_ratio = novelty_max_overlap_ratio
        self.novelty_min_pixels = novelty_min_pixels
        self.novelty_min_overlap_with_last = novelty_min_overlap_with_last
        self.max_keyframes = max_keyframes
        self.keyframe_latest_wins = keyframe_latest_wins
        # window_merge is stored but currently degrades to GPS-only
        self.use_window_merge = use_window_merge
        self.compute_window_mosaics = compute_window_mosaics
        self.window_size = window_size
        self.window_stride = window_stride

    def _sample_bilinear(
        self,
        img: np.ndarray,
        sx: np.ndarray,
        sy: np.ndarray,
    ) -> np.ndarray:
        """
        Sample RGB values with bilinear interpolation.

        Using bilinear sampling reduces jagged seams compared with the previous
        nearest-neighbour lookup, especially after the homography warp.
        """
        x0 = np.floor(sx).astype(np.int32)
        y0 = np.floor(sy).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)

        wx = (sx - x0).astype(np.float32)
        wy = (sy - y0).astype(np.float32)

        top_left = img[y0, x0].astype(np.float32)
        top_right = img[y0, x1].astype(np.float32)
        bottom_left = img[y1, x0].astype(np.float32)
        bottom_right = img[y1, x1].astype(np.float32)

        top = top_left * (1.0 - wx[:, None]) + top_right * wx[:, None]
        bottom = bottom_left * (1.0 - wx[:, None]) + bottom_right * wx[:, None]
        return top * (1.0 - wy[:, None]) + bottom * wy[:, None]

    @staticmethod
    def _image_canvas_bbox(
        H_norm: np.ndarray, H_canvas: int, W_canvas: int,
    ) -> tuple[int, int, int, int]:
        """Row/col range of the canvas that can contain this image (with margin)."""
        corners = H_norm @ np.array([[0.0, 1.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0, 1.0]])
        full = (0, H_canvas, 0, W_canvas)
        if not np.all(np.isfinite(corners)) or np.any(corners[2] <= 1e-9):
            return full  # degenerate / behind the camera: scan everything
        xs = corners[0] / corners[2] * max(W_canvas - 1, 1)
        ys = corners[1] / corners[2] * max(H_canvas - 1, 1)
        col0 = max(0, int(np.floor(xs.min())) - 2)
        col1 = min(W_canvas, int(np.ceil(xs.max())) + 3)
        row0 = max(0, int(np.floor(ys.min())) - 2)
        row1 = min(H_canvas, int(np.ceil(ys.max())) + 3)
        return row0, row1, col0, col1

    def _canvas_strips(self, H_norm: np.ndarray, H_canvas: int, W_canvas: int):
        """
        Inverse-warp one image onto the canvas a strip of rows at a time.

        Yields ``(canvas_y, canvas_x, src_u, src_v)`` for the canvas pixels that
        fall inside the image. Only the image's bounding box is visited and the
        temporaries are limited to ~STRIP_PIXELS pixels, so peak memory no longer
        scales with the canvas size.
        """
        row0, row1, col0, col1 = self._image_canvas_bbox(H_norm, H_canvas, W_canvas)
        width = col1 - col0
        if width <= 0 or row1 <= row0:
            return
        H_inv = np.linalg.inv(H_norm)
        cols = np.arange(col0, col1)
        cu_row = cols / max(W_canvas - 1, 1)
        rows_per_strip = max(1, STRIP_PIXELS // width)

        for r0 in range(row0, row1, rows_per_strip):
            r1 = min(r0 + rows_per_strip, row1)
            rows = np.arange(r0, r1)
            cu = np.tile(cu_row, r1 - r0)
            cv = np.repeat(rows / max(H_canvas - 1, 1), width)
            src_h = H_inv @ np.vstack([cu, cv, np.ones_like(cu)])
            w3 = src_h[2]
            safe_w3 = np.where(np.abs(w3) > 1e-12, w3, np.finfo(float).eps)
            src_u = src_h[0] / safe_w3
            src_v = src_h[1] / safe_w3
            valid = (
                (src_u >= 0.0) & (src_u <= 1.0) &
                (src_v >= 0.0) & (src_v <= 1.0)
            )
            if not np.any(valid):
                continue
            yield (
                np.repeat(rows, width)[valid],
                np.tile(cols, r1 - r0)[valid],
                src_u[valid],
                src_v[valid],
            )

    def _feather_weights(
        self,
        src_u: np.ndarray,
        src_v: np.ndarray,
    ) -> np.ndarray:
        """
        Prefer the centre of each image over its borders when overlaps occur.

        This reduces hard seams from edge distortions and slight GPS mismatch
        while keeping the same GPS-guided geometry.
        """
        border_dist = np.minimum.reduce([
            src_u,
            src_v,
            1.0 - src_u,
            1.0 - src_v,
        ])
        weights = np.clip(border_dist / 0.20, 0.0, 1.0)
        return np.maximum(weights, 0.05).astype(np.float32)

    def _sharpen_canvas(
        self,
        canvas: np.ndarray,
        painted_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply a mild unsharp mask to recover detail softened by bilinear warp.

        The kernel is intentionally small and conservative so it improves local
        contrast without introducing strong halos or changing empty regions.
        """
        if not np.any(painted_mask):
            return canvas

        canvas_f = canvas.astype(np.float32)
        padded = np.pad(canvas_f, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blurred = (
            4.0 * padded[1:-1, 1:-1]
            + 2.0 * (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            )
            + (
                padded[:-2, :-2]
                + padded[:-2, 2:]
                + padded[2:, :-2]
                + padded[2:, 2:]
            )
        ) / 16.0

        sharpened = np.clip(canvas_f * 1.22 - blurred * 0.22, 0, 255).astype(np.uint8)
        result = canvas.copy()
        result[painted_mask] = sharpened[painted_mask]
        return result

    def _compute_homography(
        self,
        src_footprint: FootprintResult,
        dst_footprint: FootprintResult,
    ) -> np.ndarray:
        """
        Compute the 3×3 homography mapping src_footprint's normalised [0,1]²
        coordinate space to dst_footprint's normalised [0,1]² space.

        Convention for both footprints:
            TL corner = (u=0, v=0),  TR = (u=1, v=0),
            BR = (u=1, v=1),         BL = (u=0, v=1)
        where u increases eastward and v increases southward (top-of-image = 0).

        The mapping is GPS-guided: each GPS corner of src_footprint is located
        within dst_footprint's axis-aligned bounding box to derive the
        corresponding destination normalised position.

        Parameters
        ----------
        src_footprint : object with .corners: list[GPSCoordinate] (TL,TR,BR,BL)
        dst_footprint : object with .corners: list[GPSCoordinate] (TL,TR,BR,BL)

        Returns
        -------
        np.ndarray  shape (3, 3)
        """
        # Src corners in normalised [0,1]²: TL(0,0) TR(1,0) BR(1,1) BL(0,1)
        src_norm = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )

        # Derive dst bounding box from its GPS corners (north-up, axis-aligned)
        dst_lats = [c.lat for c in dst_footprint.corners]
        dst_lons = [c.lon for c in dst_footprint.corners]
        lat_max = max(dst_lats)
        lat_min = min(dst_lats)
        lon_max = max(dst_lons)
        lon_min = min(dst_lons)
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        # Map each GPS corner of src into dst normalised space
        dst_norm = np.array(
            [
                [
                    (c.lon - lon_min) / lon_span,   # u: longitude axis
                    (lat_max - c.lat) / lat_span,   # v: latitude axis (top = 0)
                ]
                for c in src_footprint.corners
            ],
            dtype=np.float64,
        )

        return _dlt_homography(src_norm, dst_norm)

    def _compute_refined_placements(self, canvas_fp) -> list[np.ndarray]:
        """
        Delegate to alignment_refiner.refine_placements().

        Kept for backward compatibility with code that calls this directly.
        Returns refined homographies only (metrics are discarded).
        """
        from alignment_refiner import refine_placements
        h_gps = [self._compute_homography(fp, canvas_fp) for fp in self.footprints]
        refined, _ = refine_placements(self.images, h_gps)
        return refined

    # ------------------------------------------------------------------
    # Novelty-based frame filter
    # ------------------------------------------------------------------

    @staticmethod
    def _footprint_bbox(fp) -> tuple[float, float, float, float]:
        """Return an axis-aligned footprint bbox as (lat_min, lat_max, lon_min, lon_max)."""
        lats = [c.lat for c in fp.corners]
        lons = [c.lon for c in fp.corners]
        return min(lats), max(lats), min(lons), max(lons)

    @staticmethod
    def _heading_from_footprint(fp) -> float:
        """Approximate image heading in degrees from the top edge of the footprint."""
        tl = fp.corners[0]
        tr = fp.corners[1]
        avg_lat_r = math.radians((tl.lat + tr.lat) / 2.0)
        dlon = (tr.lon - tl.lon) * math.cos(avg_lat_r)
        dlat = tr.lat - tl.lat
        return math.degrees(math.atan2(dlon, dlat)) % 360.0

    @staticmethod
    def _heading_delta_deg(a: float, b: float) -> float:
        """Return the smallest absolute angular difference in degrees."""
        diff = abs(a - b) % 360.0
        return diff if diff <= 180.0 else 360.0 - diff

    @staticmethod
    def _gps_distance_m(fp_a, fp_b) -> float:
        """Approximate nadir-to-nadir distance in metres."""
        metres_per_lat = 111_320.0
        lat_a = fp_a.nadir.lat
        lon_a = fp_a.nadir.lon
        lat_b = fp_b.nadir.lat
        lon_b = fp_b.nadir.lon
        avg_lat_r = math.radians((lat_a + lat_b) / 2.0)
        dlat_m = (lat_b - lat_a) * metres_per_lat
        dlon_m = (lon_b - lon_a) * metres_per_lat * math.cos(avg_lat_r)
        return math.hypot(dlat_m, dlon_m)

    @staticmethod
    def _downsample_gray(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize to a small grayscale image using numpy-only nearest sampling."""
        h_img, w_img = img.shape[:2]
        ys = np.linspace(0, h_img - 1, max(1, target_h)).astype(np.int32)
        xs = np.linspace(0, w_img - 1, max(1, target_w)).astype(np.int32)
        sampled = img[ys][:, xs]
        if sampled.ndim == 3:
            return (
                0.299 * sampled[:, :, 0].astype(np.float32)
                + 0.587 * sampled[:, :, 1].astype(np.float32)
                + 0.114 * sampled[:, :, 2].astype(np.float32)
            )
        return sampled.astype(np.float32)

    @classmethod
    def _visual_overlap_metrics(
        cls,
        img_a: np.ndarray,
        fp_a,
        img_b: np.ndarray,
        fp_b,
    ) -> dict:
        """Compare two images inside their GPS-predicted shared footprint."""
        lat_min_a, lat_max_a, lon_min_a, lon_max_a = cls._footprint_bbox(fp_a)
        lat_min_b, lat_max_b, lon_min_b, lon_max_b = cls._footprint_bbox(fp_b)
        lat_lo = max(lat_min_a, lat_min_b)
        lat_hi = min(lat_max_a, lat_max_b)
        lon_lo = max(lon_min_a, lon_min_b)
        lon_hi = min(lon_max_a, lon_max_b)
        if lat_hi <= lat_lo or lon_hi <= lon_lo:
            return {"comparable": False, "overlap_ratio": 0.0, "similarity": 0.0, "mean_abs_diff": 1.0}

        area_b = max((lat_max_b - lat_min_b) * (lon_max_b - lon_min_b), 1e-12)
        overlap_ratio = float(((lat_hi - lat_lo) * (lon_hi - lon_lo)) / area_b)

        def _crop_bounds(img: np.ndarray, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
            h_img, w_img = img.shape[:2]
            u0 = (lon_lo - lon_min) / max(lon_max - lon_min, 1e-12)
            u1 = (lon_hi - lon_min) / max(lon_max - lon_min, 1e-12)
            v0 = (lat_max - lat_hi) / max(lat_max - lat_min, 1e-12)
            v1 = (lat_max - lat_lo) / max(lat_max - lat_min, 1e-12)
            x0 = int(np.clip(np.floor(u0 * w_img), 0, max(w_img - 1, 0)))
            x1 = int(np.clip(np.ceil(u1 * w_img), x0 + 1, w_img))
            y0 = int(np.clip(np.floor(v0 * h_img), 0, max(h_img - 1, 0)))
            y1 = int(np.clip(np.ceil(v1 * h_img), y0 + 1, h_img))
            return x0, x1, y0, y1

        x0a, x1a, y0a, y1a = _crop_bounds(img_a, lat_min_a, lat_max_a, lon_min_a, lon_max_a)
        x0b, x1b, y0b, y1b = _crop_bounds(img_b, lat_min_b, lat_max_b, lon_min_b, lon_max_b)
        crop_a = img_a[y0a:y1a, x0a:x1a]
        crop_b = img_b[y0b:y1b, x0b:x1b]
        if crop_a.size == 0 or crop_b.size == 0 or min(crop_a.shape[:2]) < 8 or min(crop_b.shape[:2]) < 8:
            return {"comparable": False, "overlap_ratio": overlap_ratio, "similarity": 0.0, "mean_abs_diff": 1.0}

        target_h = min(96, crop_a.shape[0], crop_b.shape[0])
        target_w = min(96, crop_a.shape[1], crop_b.shape[1])
        gray_a = cls._downsample_gray(crop_a, target_h, target_w)
        gray_b = cls._downsample_gray(crop_b, target_h, target_w)
        mad = float(np.mean(np.abs(gray_a - gray_b)) / 255.0)
        flat_a = gray_a.astype(np.float64).ravel()
        flat_b = gray_b.astype(np.float64).ravel()
        std_a = float(np.std(flat_a))
        std_b = float(np.std(flat_b))
        if std_a < 1e-6 or std_b < 1e-6:
            corr01 = 1.0 if mad < 0.02 else max(0.0, 1.0 - mad)
        else:
            corr = float(np.corrcoef(flat_a, flat_b)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
            corr01 = 0.5 * (corr + 1.0)
        similarity = float(np.clip(0.65 * corr01 + 0.35 * max(0.0, 1.0 - mad), 0.0, 1.0))
        return {
            "comparable": True,
            "overlap_ratio": overlap_ratio,
            "similarity": similarity,
            "mean_abs_diff": mad,
        }

    def _filter_novelty_frames(self) -> tuple[list[np.ndarray], list, dict]:
        """
        Select a subset of frames that contribute novel content.

        Uses GPS, image angle, overlap geometry, overlap similarity, and
        estimated new edge pixels. A frame is filtered only when it is both
        geometrically redundant and visually redundant.

        **Global comparison:** each frame is compared against up to K=3 nearest
        kept frames (by GPS distance), not just the last kept frame. This catches
        cross-strip near-duplicates that would otherwise slip through.

        Returns (filtered_images, filtered_footprints, filter_metrics).
        """
        images_out: list[np.ndarray] = []
        fps_out: list = []
        skipped = 0
        duplicate_filtered = 0
        edge_kept = 0
        visual_kept = 0
        angle_kept = 0
        gps_kept = 0
        novelty_decisions: list[dict] = []
        kept_imgs: list[np.ndarray] = []
        kept_fps: list = []
        kept_last_heading: float = 0.0
        similarity_duplicate_threshold = 0.94
        _COMPARE_K = 3  # compare against K nearest kept frames

        for frame_idx, (img, fp) in enumerate(zip(self.images, self.footprints)):
            heading = self._heading_from_footprint(fp)

            # First frame: always keep
            if not kept_fps:
                images_out.append(img)
                fps_out.append(fp)
                kept_imgs.append(img)
                kept_fps.append(fp)
                kept_last_heading = heading
                novelty_decisions.append({
                    "decision": "keep_first",
                    "frame_idx": frame_idx,
                    "compared_to": [],
                })
                continue

            # Find K nearest kept frames by GPS distance
            dists_to_kept = [self._gps_distance_m(kfp, fp) for kfp in kept_fps]
            sorted_kept_idx = sorted(range(len(dists_to_kept)), key=lambda x: dists_to_kept[x])
            close_kept_idx = sorted_kept_idx[:_COMPARE_K]
            dist_m = dists_to_kept[close_kept_idx[0]] if close_kept_idx else 0.0

            # Compare against all close neighbours; take worst-case (maximum overlap)
            angle_diff = self._heading_delta_deg(heading, kept_last_heading)
            best_visual = None
            max_overlap_ratio = 0.0
            for ci in close_kept_idx:
                v = self._visual_overlap_metrics(kept_imgs[ci], kept_fps[ci], img, fp)
                if v["overlap_ratio"] > max_overlap_ratio:
                    max_overlap_ratio = v["overlap_ratio"]
                    best_visual = v
            if best_visual is None:
                best_visual = {"overlap_ratio": 0.0, "comparable": False, "similarity": 0.0, "mean_abs_diff": 1.0}
            visual = best_visual
            overlap_ratio = float(visual["overlap_ratio"])

            exclusive_pixels_est = int(round(max(0.0, 1.0 - overlap_ratio) * img.shape[0] * img.shape[1]))
            gps_novel = dist_m >= self.novelty_gps_min_move_m
            angle_novel = angle_diff >= self.novelty_angle_threshold_deg
            overlap_novel = overlap_ratio < self.novelty_max_overlap_ratio
            edge_novel = exclusive_pixels_est >= self.novelty_min_pixels
            visual_novel = (
                visual["comparable"]
                and overlap_ratio >= self.novelty_min_overlap_with_last
                and visual["similarity"] < similarity_duplicate_threshold
            )
            visually_duplicate = (not visual["comparable"]) or (
                visual["similarity"] >= similarity_duplicate_threshold
            )
            duplicate_candidate = (
                overlap_ratio >= self.novelty_max_overlap_ratio
                and exclusive_pixels_est < self.novelty_min_pixels
                and angle_diff < self.novelty_angle_threshold_deg
                and visually_duplicate
            )

            if not duplicate_candidate:
                if self.max_keyframes is None or len(images_out) < self.max_keyframes:
                    images_out.append(img)
                    fps_out.append(fp)
                    kept_imgs.append(img)
                    kept_fps.append(fp)
                    kept_last_heading = heading
                    reason = "overlap_novel"
                    if angle_novel:
                        angle_kept += 1
                        reason = "angle_novel"
                    elif edge_novel:
                        edge_kept += 1
                        reason = "edge_novel"
                    elif visual_novel:
                        visual_kept += 1
                        reason = "visual_novel"
                    elif gps_novel:
                        gps_kept += 1
                        reason = "gps_novel"
                    novelty_decisions.append({
                        "decision": "keep",
                        "frame_idx": frame_idx,
                        "compared_to": close_kept_idx,
                        "reason": reason,
                        "distance_m": round(dist_m, 3),
                        "angle_diff_deg": round(angle_diff, 3),
                        "overlap_ratio": round(overlap_ratio, 4),
                        "exclusive_pixels_est": exclusive_pixels_est,
                        "similarity": round(float(visual["similarity"]), 4),
                    })
                else:
                    skipped += 1
                    novelty_decisions.append({
                        "decision": "skip",
                        "frame_idx": frame_idx,
                        "compared_to": close_kept_idx,
                        "reason": "max_keyframes",
                    })
            else:
                skipped += 1
                duplicate_filtered += 1
                novelty_decisions.append({
                    "decision": "skip",
                    "frame_idx": frame_idx,
                    "compared_to": close_kept_idx,
                    "reason": "duplicate_frame",
                    "distance_m": round(dist_m, 3),
                    "angle_diff_deg": round(angle_diff, 3),
                    "overlap_ratio": round(overlap_ratio, 4),
                    "exclusive_pixels_est": exclusive_pixels_est,
                    "similarity": round(float(visual["similarity"]), 4),
                })

        metrics = {
            "frames_input": len(self.images),
            "frames_used": len(images_out),
            "frames_filtered": skipped,
            "duplicate_frames_filtered": duplicate_filtered,
            "edge_novel_frames_kept": edge_kept,
            "visual_novel_frames_kept": visual_kept,
            "angle_novel_frames_kept": angle_kept,
            "gps_novel_frames_kept": gps_kept,
            "novelty_decisions": novelty_decisions,
        }
        return images_out, fps_out, metrics

    def stitch(self) -> MosaicResult:
        """Stitch the mosaic (occupies the 40-95 % slice of a render job)."""
        with progress.window(0.4, 0.95):
            return self._stitch_impl()

    def _stitch_impl(self) -> MosaicResult:
        """
        GPS-guided inverse-warp mosaic with optional advanced pipeline stages.

        Pipeline
        --------
        0. Novelty filter (if use_novelty_composite): drop redundant frames.
        1. Compute canvas GPS bounds from footprint corners.
        2. GSD = minimum per-image GSD (finest resolution wins).
        3. Allocate north-up canvas.
        4. Compute GPS homographies for every image.
        5. Refine homographies (if use_alignment_refinement or refine):
             alignment_refiner.refine_placements() — ORB consecutive matching.
        6. Structural optimization (if use_sba):
             structural_optimizer.optimize_poses() — global drift correction.
        7. Inverse-warp each image onto canvas with bilinear sampling.
        8. Composite using the requested mode:
             smart_blend        — feather-weighted average (default advanced)
             sharpest           — per-pixel sharpest-source selection
             seamline           — Voronoi seam-cut (if use_semantic_seamline)
             last_painter_wins  — (legacy baseline, not exposed via API flags)
        9. Post-sharpen.

        Returns
        -------
        MosaicResult  (always has .diagnostics dict populated)

        Raises
        ------
        ValueError  If no images are provided or image/footprint counts mismatch.
        """
        if not self.images:
            raise ValueError("No images provided to stitch.")
        if len(self.images) != len(self.footprints):
            raise ValueError(
                f"Image count ({len(self.images)}) must match "
                f"footprint count ({len(self.footprints)})."
            )

        diagnostics: dict = {}

        # ------------------------------------------------------------------
        # 0. Novelty filter
        # ------------------------------------------------------------------
        if self.use_novelty_composite and len(self.images) > 1:
            images, footprints, nf_metrics = self._filter_novelty_frames()
            diagnostics.update(nf_metrics)
            if len(images) < 1:
                # Filter removed everything — fall back to all images
                images, footprints = self.images, self.footprints
                diagnostics["novelty_fallback"] = "all_filtered_reverted"
        else:
            images, footprints = self.images, self.footprints
            diagnostics["frames_input"] = len(self.images)
            diagnostics["frames_used"] = len(self.images)
            diagnostics["frames_filtered"] = 0

        # ------------------------------------------------------------------
        progress.report(0.02, "Selecting frames")
        # 1. Canvas GPS bounds  (use all original footprints for stable bounds)
        # ------------------------------------------------------------------
        all_lats: list[float] = []
        all_lons: list[float] = []
        for fp in self.footprints:   # always ALL original frames, not the filtered subset
            for c in fp.corners:
                all_lats.append(c.lat)
                all_lons.append(c.lon)
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)

        # ------------------------------------------------------------------
        # 2. GSD: finest resolution across used images
        # ------------------------------------------------------------------
        METRES_PER_LAT_DEG: float = 111_320.0
        gsd_list: list[float] = []
        for img, fp in zip(images, footprints):
            w_px = max(img.shape[1], 1)
            gsd_list.append(fp.ground_width_m / w_px)
        gsd_m = min(gsd_list)

        # ------------------------------------------------------------------
        # 3. Canvas pixel dimensions
        # ------------------------------------------------------------------
        avg_lat_rad = math.radians((lat_min + lat_max) / 2.0)
        lat_span_m = (lat_max - lat_min) * METRES_PER_LAT_DEG
        lon_span_m = (lon_max - lon_min) * METRES_PER_LAT_DEG * math.cos(avg_lat_rad)
        W_canvas = max(1, round(lon_span_m / gsd_m))
        H_canvas = max(1, round(lat_span_m / gsd_m))

        # Memory guard: the compositor allocates several float arrays of
        # canvas size, so very fine GSDs over large areas are coarsened.
        canvas_downscaled = False
        if W_canvas * H_canvas > MAX_CANVAS_PIXELS:
            shrink = math.sqrt(MAX_CANVAS_PIXELS / (W_canvas * H_canvas))
            W_canvas = max(1, int(W_canvas * shrink))
            H_canvas = max(1, int(H_canvas * shrink))
            gsd_m = gsd_m / shrink
            canvas_downscaled = True

        # ------------------------------------------------------------------
        # 4. Canvas footprint descriptor (duck-typed _CanvasFP)
        # ------------------------------------------------------------------
        class _CanvasFP:
            corners = [
                GPSCoordinate(lat=lat_max, lon=lon_min),  # TL: NW
                GPSCoordinate(lat=lat_max, lon=lon_max),  # TR: NE
                GPSCoordinate(lat=lat_min, lon=lon_max),  # BR: SE
                GPSCoordinate(lat=lat_min, lon=lon_min),  # BL: SW
            ]

        canvas_fp = _CanvasFP()

        # ------------------------------------------------------------------
        # 5. GPS baseline homographies
        # ------------------------------------------------------------------
        h_gps = [self._compute_homography(fp, canvas_fp) for fp in footprints]

        # ------------------------------------------------------------------
        # 5a. Alignment refinement (ORB) — optional
        # ------------------------------------------------------------------
        align_metrics: dict = {}
        if self.use_alignment_refinement and len(images) > 1:
            from alignment_refiner import refine_placements
            h_norms, align_metrics = refine_placements(images, h_gps)
        else:
            h_norms = h_gps

        diagnostics["alignment"] = align_metrics
        diagnostics["alignment_applied"] = bool(
            self.use_alignment_refinement and align_metrics.get("accepted_pairs", 0) > 0
        )
        # Surface key alignment scalars at the top level for easy API consumption
        diagnostics["accepted_pairs"] = align_metrics.get("accepted_pairs", 0)
        diagnostics["rejected_pairs"] = align_metrics.get("rejected_pairs", 0)
        diagnostics["mean_inliers"] = align_metrics.get("mean_inliers", 0.0)

        # ------------------------------------------------------------------
        # 5b. Structural bundle adjustment — optional
        # ------------------------------------------------------------------
        sba_meta: dict = {}
        if self.use_sba and len(images) > 2:
            try:
                from structural_optimizer import optimize_poses
                # Forward non-sequential visual edges from the refiner so the
                # optimizer uses the full pose graph, not just the sequential chain.
                visual_edges_for_sba = align_metrics.get("visual_edges", None)
                h_norms, sba_meta = optimize_poses(
                    h_gps, h_norms,
                    gps_weight=self.sba_weights.get("w1", 0.4),
                    orb_weight=self.sba_weights.get("w2", 1.0),
                    visual_edges=visual_edges_for_sba,
                )
            except Exception as exc:
                sba_meta = {"error": str(exc), "correction_applied": False}

        diagnostics["sba"] = sba_meta
        diagnostics["sba_applied"] = bool(sba_meta.get("correction_applied", False))
        # Surface per-image pose error from the optimizer
        diagnostics["gps_anchor_error_px"] = sba_meta.get("gps_anchor_error_px", 0.0)
        diagnostics["refinement_shift_px"] = sba_meta.get("refinement_shift_px", 0.0)

        # ------------------------------------------------------------------
        progress.report(0.12, "Aligning images")
        # 6. (canvas coordinates are generated per row-strip, see _canvas_strips)
        # ------------------------------------------------------------------
        # 7 + 8. Inverse warp + composite
        # ------------------------------------------------------------------
        use_seamline = self.use_semantic_seamline
        use_sharpest = (self.composite_mode == "sharpest") and not use_seamline

        # Shared accumulators for smart_blend / seamline paths
        accum = np.zeros((H_canvas, W_canvas, 3), dtype=np.float32)
        weights_sum = np.zeros((H_canvas, W_canvas), dtype=np.float32)

        # Seamline contributions list (populated only when use_seamline)
        from seamline_detector import ImageContribution, SeamlineBlender
        seamline_contributions: list[ImageContribution] = []

        # Sharpest compositor (populated only when use_sharpest)
        from render_refined_mosaic import (
            SharpestCompositor, compute_image_sharpness,
        )
        sharpest_comp: Optional[SharpestCompositor] = (
            SharpestCompositor(H_canvas, W_canvas) if use_sharpest else None
        )

        for _idx, (img, H_norm) in enumerate(zip(images, h_norms)):
            progress.report(0.2 + 0.7 * _idx / max(len(images), 1), f"Compositing image {_idx + 1}/{len(images)}")
            h_img, w_img = img.shape[:2]

            seam_parts: list[tuple] = []
            sharpness = compute_image_sharpness(img) if use_sharpest else 0.0

            for v_y, v_x, su, sv in self._canvas_strips(H_norm, H_canvas, W_canvas):
                sx = su * (w_img - 1)
                sy = sv * (h_img - 1)
                sampled = self._sample_bilinear(img, sx, sy)

                if use_seamline:
                    seam_parts.append((sampled.astype(np.float32), su.astype(np.float32),
                                       sv.astype(np.float32), v_y, v_x))
                    continue

                feather_w = self._feather_weights(su, sv)
                if use_sharpest:
                    sharpest_comp.update(v_y, v_x, sampled.astype(np.float32),
                                         feather_w, sharpness)
                else:
                    # smart_blend (feather-weighted average)
                    accum[v_y, v_x] += sampled * feather_w[:, None]
                    weights_sum[v_y, v_x] += feather_w

            if use_seamline and seam_parts:
                # Map the image centre (0.5, 0.5, 1) through H_norm to the canvas
                c_h = H_norm @ np.array([0.5, 0.5, 1.0])
                cw3 = c_h[2] if abs(c_h[2]) > 1e-12 else 1e-12
                c_cu = float(c_h[0] / cw3)
                c_cv = float(c_h[1] / cw3)
                seamline_contributions.append(ImageContribution(
                    sampled_rgb=np.concatenate([s[0] for s in seam_parts]),
                    src_u=np.concatenate([s[1] for s in seam_parts]),
                    src_v=np.concatenate([s[2] for s in seam_parts]),
                    canvas_y=np.concatenate([s[3] for s in seam_parts]),
                    canvas_x=np.concatenate([s[4] for s in seam_parts]),
                    centre_u=float(np.clip(c_cu, 0.0, 1.0)),
                    centre_v=float(np.clip(c_cv, 0.0, 1.0)),
                ))

        # ------------------------------------------------------------------
        progress.report(0.92, "Blending")
        # 8. Final compositing pass
        # ------------------------------------------------------------------
        if use_seamline:
            blender = SeamlineBlender(H_canvas, W_canvas)
            canvas = blender.blend(seamline_contributions)
            canvas = self._sharpen_canvas(canvas, np.any(canvas != 0, axis=2))
            diagnostics["compositing_mode"] = "seamline_voronoi"
        elif use_sharpest:
            canvas = sharpest_comp.result(post_sharpen=True)
            diagnostics["compositing_mode"] = "sharpest"
        else:
            from render_refined_mosaic import smart_blend as _smart_blend
            canvas = _smart_blend(accum, weights_sum, post_sharpen=True)
            diagnostics["compositing_mode"] = "smart_blend"

        # ------------------------------------------------------------------
        # 9. Canvas coverage metric
        # ------------------------------------------------------------------
        painted_mask = np.any(canvas != 0, axis=2)
        total_px = H_canvas * W_canvas
        diagnostics["canvas_coverage_pct"] = round(
            float(painted_mask.sum()) / max(total_px, 1) * 100.0, 1
        )
        diagnostics["canvas_width_px"] = W_canvas
        diagnostics["canvas_height_px"] = H_canvas
        diagnostics["gsd_m"] = round(gsd_m, 6)
        diagnostics["canvas_downscaled"] = canvas_downscaled
        diagnostics["seamline_applied"] = use_seamline

        bounding_box = [
            GPSCoordinate(lat=lat_max, lon=lon_min),
            GPSCoordinate(lat=lat_max, lon=lon_max),
            GPSCoordinate(lat=lat_min, lon=lon_max),
            GPSCoordinate(lat=lat_min, lon=lon_min),
        ]

        return MosaicResult(
            image=canvas,
            bounding_box=bounding_box,
            gsd_m=gsd_m,
            diagnostics=diagnostics,
        )


# ===========================================================================
# Task 4 – Image-to-GPS Projection
# ===========================================================================

class GPSProjector:
    """
    Task 4: Converts pixel coordinates in a mosaic image to GPS coordinates.

    This is the inverse of the footprint projection: given a pixel (u, v) in
    the stitched mosaic, return the corresponding GPS position on the ground.

    Inputs
    ------
    mosaic : MosaicResult  — stitched image with its bounding box (Task 3)

    Output
    ------
    GPSCoordinate
    """

    def __init__(self, mosaic: MosaicResult) -> None:
        self.mosaic = mosaic

    def pixel_to_gps(self, pixel_x: int, pixel_y: int) -> GPSCoordinate:
        """
        Map a pixel location in the mosaic to a GPS coordinate.

        Parameters
        ----------
        pixel_x : int   Horizontal pixel index (0 = left).
        pixel_y : int   Vertical   pixel index (0 = top).

        Returns
        -------
        GPSCoordinate   The corresponding ground GPS position.

        Raises
        ------
        IndexError  If the pixel is outside the image bounds.
        """
        h, w = self.mosaic.image.shape[:2]
        if pixel_x < 0 or pixel_x >= w or pixel_y < 0 or pixel_y >= h:
            raise IndexError(
                f"Pixel ({pixel_x}, {pixel_y}) is out of range for image {w}×{h}."
            )
        lats = [c.lat for c in self.mosaic.bounding_box]
        lons = [c.lon for c in self.mosaic.bounding_box]
        lat_max, lat_min = max(lats), min(lats)
        lon_max, lon_min = max(lons), min(lons)
        # Linear map: pixel 0 → min edge, pixel (dim-1) → max edge
        lon = lon_min + (pixel_x / (w - 1)) * (lon_max - lon_min)
        lat = lat_max - (pixel_y / (h - 1)) * (lat_max - lat_min)
        return GPSCoordinate(lat=lat, lon=lon)

    def gps_to_pixel(self, coord: GPSCoordinate) -> tuple[int, int]:
        """
        Map a GPS coordinate to a pixel location in the mosaic.

        Parameters
        ----------
        coord : GPSCoordinate

        Returns
        -------
        tuple[int, int]   (pixel_x, pixel_y)

        Raises
        ------
        ValueError  If the coordinate is outside the mosaic bounding box.
        """
        lats = [c.lat for c in self.mosaic.bounding_box]
        lons = [c.lon for c in self.mosaic.bounding_box]
        lat_max, lat_min = max(lats), min(lats)
        lon_max, lon_min = max(lons), min(lons)
        if not (lat_min <= coord.lat <= lat_max and lon_min <= coord.lon <= lon_max):
            raise ValueError(
                f"GPS ({coord.lat}, {coord.lon}) is outside the mosaic bounding box "
                f"lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]."
            )
        h, w = self.mosaic.image.shape[:2]
        pixel_x = round((coord.lon - lon_min) / (lon_max - lon_min) * (w - 1))
        pixel_y = round((lat_max - coord.lat) / (lat_max - lat_min) * (h - 1))
        return (pixel_x, pixel_y)


# ===========================================================================
# Task 5 – Altitude-Adaptive Scaling
# ===========================================================================

class AltitudeScaler:
    """
    Task 5: Adjusts image scale and GSD based on real-time altitude changes.

    When a drone does not fly at a perfectly constant altitude (terrain
    following, wind gusts, etc.), images have slightly different GSDs.
    This module normalises them to a common reference GSD.

    Inputs
    ------
    reference_altitude_m : float   — the nominal flight altitude
    reference_gsd_m      : float   — the desired GSD at the reference altitude
    camera_config        : CameraConfig

    Output
    ------
    float   — scaling factor to apply to the image before stitching
    """

    def __init__(
        self,
        reference_altitude_m: float,
        reference_gsd_m: float,
        camera_config: CameraConfig,
    ) -> None:
        self.reference_altitude_m = reference_altitude_m
        self.reference_gsd_m = reference_gsd_m
        self.camera_config = camera_config

    def compute_scale_factor(self, actual_altitude_m: float) -> float:
        """
        Compute the scale factor to normalise an image taken at *actual_altitude_m*
        to the reference GSD.

        Parameters
        ----------
        actual_altitude_m : float   AGL altitude at the moment of capture.

        Returns
        -------
        float   Multiplicative scale factor (>1 means zoom in, <1 means zoom out).
        """
        if actual_altitude_m <= 0:
            raise ValueError(f"Altitude must be positive, got {actual_altitude_m} m.")
        if self.reference_altitude_m <= 0:
            raise ValueError(
                f"Reference altitude must be positive, got {self.reference_altitude_m} m."
            )
        # GSD is proportional to altitude (nadir, flat-earth).
        # To normalise actual image to reference_gsd: scale = actual_gsd / reference_gsd
        # = actual_altitude / reference_altitude
        return actual_altitude_m / self.reference_altitude_m

    def compute_gsd(self, altitude_m: float, image_width_px: int) -> float:
        """
        Compute the ground sampling distance for a given altitude and image width.

        Parameters
        ----------
        altitude_m     : float   AGL altitude in metres.
        image_width_px : int     Sensor horizontal resolution in pixels.

        Returns
        -------
        float   GSD in metres per pixel.
        """
        if altitude_m <= 0:
            raise ValueError(f"Altitude must be positive, got {altitude_m} m.")
        if image_width_px <= 0:
            raise ValueError(f"image_width_px must be positive, got {image_width_px}.")
        ground_width_m = (
            2.0 * altitude_m * math.tan(math.radians(self.camera_config.fov_h / 2.0))
        )
        return ground_width_m / image_width_px


# MissionPlanner is now imported from mission_planner.py at the top of the file


# ===========================================================================
# DroneOversightSystem  —  Top-level orchestrator
# ===========================================================================

class DroneOversightSystem:
    """
    Top-level façade that wires all six tasks together.

    Usage
    -----
    >>> system = DroneOversightSystem(camera_config)
    >>> footprint = system.compute_footprint(telemetry)
    >>> system.heatmap.add_footprint(footprint)
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        area_corners: Optional[list[GPSCoordinate]] = None,
        grid_resolution_m: float = 10.0,
        overlap_h: float = 0.7,
        overlap_v: float = 0.7,
    ) -> None:
        self.camera_config = camera_config

        # Task 2 — initialise heatmap if area is known upfront
        self.heatmap: Optional[CoverageHeatmap] = (
            CoverageHeatmap(area_corners, grid_resolution_m)
            if area_corners
            else None
        )

        self.overlap_h = overlap_h
        self.overlap_v = overlap_v

        # Storage for stitching pipeline
        self._captured_images:   list[np.ndarray]     = []
        self._captured_footprints: list[FootprintResult] = []

    # ── Task 1 ───────────────────────────────────────────────────────────

    def compute_footprint(self, telemetry: DroneTelemetry) -> FootprintResult:
        """
        Task 1: Compute the image footprint for a given telemetry frame.

        Parameters
        ----------
        telemetry : DroneTelemetry

        Returns
        -------
        FootprintResult
        """
        calc      = FootprintCalculator(telemetry)
        footprint = calc.compute()

        # Automatically feed into heatmap (Task 2) if configured
        if self.heatmap is not None:
            self.heatmap.add_footprint(footprint)

        return footprint

    # ── Task 3 ───────────────────────────────────────────────────────────

    def add_image(self, image: np.ndarray, footprint: FootprintResult) -> None:
        """
        Register a captured image and its footprint for later stitching.

        Parameters
        ----------
        image     : np.ndarray      Raw image array.
        footprint : FootprintResult Corresponding footprint from Task 1.
        """
        self._captured_images.append(image)
        self._captured_footprints.append(footprint)

    def stitch(self) -> MosaicResult:
        """Task 3: Stitch all registered images into a geo-referenced mosaic."""
        stitcher = ImageStitcher(
            self._captured_images,
            self._captured_footprints,
            self.camera_config,
        )
        return stitcher.stitch()

    # ── Task 4 ───────────────────────────────────────────────────────────

    def project_pixel(
        self, mosaic: MosaicResult, pixel_x: int, pixel_y: int
    ) -> GPSCoordinate:
        """Task 4: Convert a mosaic pixel location to a GPS coordinate."""
        projector = GPSProjector(mosaic)
        return projector.pixel_to_gps(pixel_x, pixel_y)

    # ── Task 5 ───────────────────────────────────────────────────────────

    def get_scale_factor(
        self,
        actual_altitude_m: float,
        reference_altitude_m: float,
        reference_gsd_m: float,
    ) -> float:
        """Task 5: Compute the altitude-adaptive scale factor for an image."""
        scaler = AltitudeScaler(
            reference_altitude_m, reference_gsd_m, self.camera_config
        )
        return scaler.compute_scale_factor(actual_altitude_m)

    # ── Task 6 ───────────────────────────────────────────────────────────

    def plan_mission(
        self,
        area_polygon: list[GPSCoordinate],
        altitude_m: float,
    ) -> list[GPSCoordinate]:
        """Task 6: Generate optimal waypoints to cover the area polygon."""
        planner = MissionPlanner(
            area_polygon,
            altitude_m,
            self.camera_config,
            self.overlap_h,
            self.overlap_v,
        )
        return planner.plan()
