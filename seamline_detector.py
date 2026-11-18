"""
seamline_detector.py  —  Overlap-aware seam-cut blending for Task 3
====================================================================
Provides a ``SeamlineBlender`` that composites multiple warped image
contributions onto a shared canvas using a distance-transform Voronoi
approach:

  1. Each image's contribution is represented by the set of canvas pixels it
     covers and the source (u, v) coordinates it was sampled from.
  2. For each canvas pixel covered by more than one image, ownership is
     assigned to the image whose *projected footprint centre* is nearest (in
     canvas pixel space).  This is the Voronoi partition.
  3. Within a configurable feather band around Voronoi boundaries, the two
     nearest images are blended proportionally to their distance weight.
     Outside the feather band, the owner contributes 100%.

This produces sharper results than a pure feather-from-edge average in
non-overlap areas and smooth transitions only where images actually share
pixels — avoiding the soft-averaging blur that feather-only compositing
introduces over large uniform regions.

Pure-numpy implementation; no OpenCV required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# Width of the feather zone around Voronoi seams, expressed as a fraction
# of the smaller canvas dimension.  0.05 → ~5% of canvas height/width.
FEATHER_BAND_FRACTION: float = 0.05


@dataclass
class ImageContribution:
    """
    All per-image data the blender needs to composite one image onto the canvas.

    Attributes
    ----------
    sampled_rgb  : (K, 3) float32  — RGB values already sampled from the source.
    src_u        : (K,)   float32  — normalised u coords in source image [0, 1].
    src_v        : (K,)   float32  — normalised v coords in source image [0, 1].
    canvas_y     : (K,)   int32    — canvas row indices for each sample.
    canvas_x     : (K,)   int32    — canvas column indices for each sample.
    centre_u     : float           — projected image centre, canvas normalised u.
    centre_v     : float           — projected image centre, canvas normalised v.
    """

    sampled_rgb: np.ndarray    # (K, 3) float32
    src_u: np.ndarray          # (K,)   float32
    src_v: np.ndarray          # (K,)   float32
    canvas_y: np.ndarray       # (K,)   int32
    canvas_x: np.ndarray       # (K,)   int32
    centre_u: float = 0.5
    centre_v: float = 0.5


class SeamlineBlender:
    """
    Voronoi-seam compositor for multi-image drone mosaics.

    Parameters
    ----------
    canvas_h : int   Canvas height in pixels.
    canvas_w : int   Canvas width in pixels.
    feather_band : float
        Width of the transition zone as a fraction of ``min(H, W)``.
        Defaults to ``FEATHER_BAND_FRACTION`` (0.05).
    """

    def __init__(
        self,
        canvas_h: int,
        canvas_w: int,
        feather_band: float = FEATHER_BAND_FRACTION,
    ) -> None:
        self.H = canvas_h
        self.W = canvas_w
        self.feather_px = max(1.0, feather_band * min(canvas_h, canvas_w))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def blend(self, contributions: list[ImageContribution]) -> np.ndarray:
        """
        Composite all image contributions into a single canvas.

        Parameters
        ----------
        contributions : list[ImageContribution]
            One entry per image.  Images with zero valid pixels are silently
            skipped.

        Returns
        -------
        np.ndarray  (H, W, 3) uint8 blended canvas.
        """
        if not contributions:
            return np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # ------------------------------------------------------------------
        # Step 1: For every canvas pixel, record which images contribute and
        #         compute per-image feather weights (distance from source edges).
        # ------------------------------------------------------------------
        # Use a dict[flat_idx → list[(image_idx, rgb, feather_weight)]] to
        # accumulate contributions.  For large canvases this is memory-heavy;
        # the practical canvas sizes at preview scale (scale ≤ 0.20) stay in
        # tens of MB so this is acceptable.
        accum: np.ndarray = np.zeros((self.H, self.W, 3), dtype=np.float32)
        weight_sum: np.ndarray = np.zeros((self.H, self.W), dtype=np.float32)

        # Also track: for each canvas pixel, list of (contributor_idx, dist_to_centre)
        # We need this for the Voronoi seam decision.  We build a compact structure:
        # owner_map[y, x] = index of dominant contributor (-1 = unpainted)
        # dist_map[y, x]  = distance to owner centre (canvas pixel coords)
        owner_map = np.full((self.H, self.W), -1, dtype=np.int32)
        dist_map = np.full((self.H, self.W), np.inf, dtype=np.float32)

        # Per-image buffer: (canvas_y, canvas_x, rgb, feather_w, centre_dist)
        per_image: list[tuple] = []

        for img_idx, contrib in enumerate(contributions):
            if contrib.canvas_y.size == 0:
                per_image.append(None)
                continue

            # Feather weight: prefer source centre over edges
            border_dist = np.minimum.reduce([
                contrib.src_u,
                contrib.src_v,
                1.0 - contrib.src_u,
                1.0 - contrib.src_v,
            ])
            feather_w = np.clip(border_dist / 0.20, 0.0, 1.0)
            feather_w = np.maximum(feather_w, 0.05).astype(np.float32)

            # Distance from this image's projected centre (canvas pixels)
            centre_px_x = contrib.centre_u * (self.W - 1)
            centre_px_y = contrib.centre_v * (self.H - 1)
            dx = contrib.canvas_x.astype(np.float32) - centre_px_x
            dy = contrib.canvas_y.astype(np.float32) - centre_px_y
            centre_dist = np.sqrt(dx * dx + dy * dy)

            per_image.append((
                contrib.canvas_y,
                contrib.canvas_x,
                contrib.sampled_rgb,
                feather_w,
                centre_dist,
            ))

            # Update owner_map: nearest projected centre wins
            cy, cx = contrib.canvas_y, contrib.canvas_x
            current_owner_dist = dist_map[cy, cx]
            closer = centre_dist < current_owner_dist
            owner_map[cy[closer], cx[closer]] = img_idx
            dist_map[cy[closer], cx[closer]] = centre_dist[closer]

        # ------------------------------------------------------------------
        # Step 2: Composite using Voronoi ownership + feather at boundaries.
        # ------------------------------------------------------------------
        for img_idx, buf in enumerate(per_image):
            if buf is None:
                continue
            cy, cx, rgb, feather_w, centre_dist = buf

            # Ownership mask for this image at these pixels
            is_owner = (owner_map[cy, cx] == img_idx)

            # Blend weight:
            #  - owner pixels:     full feather weight (no sharing at seam)
            #  - non-owner pixels: feather weight scaled down by how far we
            #                       are inside the feather band around the seam
            owner_dist = dist_map[cy, cx]   # distance of the owner centre
            # dist_to_seam ≈ |my_centre_dist - owner_dist| / 2
            dist_diff = np.abs(centre_dist - owner_dist)
            seam_fade = np.clip(dist_diff / max(self.feather_px, 1.0), 0.0, 1.0)

            # Owner: weight = feather_w (high near image centre)
            # Non-owner inside feather band: weight = feather_w * (1 - seam_fade)
            weight = np.where(is_owner, feather_w, feather_w * (1.0 - seam_fade))
            weight = np.maximum(weight, 0.0).astype(np.float32)

            # Only contribute where weight > 0
            positive = weight > 1e-6
            if not np.any(positive):
                continue

            accum[cy[positive], cx[positive]] += (
                rgb[positive].astype(np.float32) * weight[positive, None]
            )
            weight_sum[cy[positive], cx[positive]] += weight[positive]

        # ------------------------------------------------------------------
        # Step 3: Normalise and return.
        # ------------------------------------------------------------------
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        painted = weight_sum > 0
        canvas[painted] = np.clip(
            accum[painted] / weight_sum[painted, None], 0, 255
        ).astype(np.uint8)
        return canvas

    # ------------------------------------------------------------------
    # Convenience: feather-only blend (no Voronoi seam cut)
    # ------------------------------------------------------------------

    @staticmethod
    def feather_blend(
        accum: np.ndarray,
        weights_sum: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a weighted-sum accumulator + weight map to a uint8 canvas.

        This is the same compositing step used by the baseline GPS path but
        exposed here as a named utility for reuse.

        Parameters
        ----------
        accum       : (H, W, 3) float32
        weights_sum : (H, W)    float32

        Returns
        -------
        np.ndarray  (H, W, 3) uint8
        """
        canvas = np.zeros((*weights_sum.shape, 3), dtype=np.uint8)
        painted = weights_sum > 0
        canvas[painted] = np.clip(
            accum[painted] / weights_sum[painted, None], 0, 255
        ).astype(np.uint8)
        return canvas
