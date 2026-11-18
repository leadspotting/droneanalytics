"""
render_refined_mosaic.py  —  Advanced compositing for Task 3
=============================================================
Provides the final render step of the advanced stitching pipeline,
after canvas placements (homographies) have been computed.

Compositing modes
-----------------
smart_blend  (default for advanced path)
    Feather-weighted average: each image contributes proportionally to how far
    its sampled point is from the image boundary.  Strongly prefers the image
    centre; edges are de-weighted.  Multiple overlapping images are blended
    smoothly with no hard seams.

sharpest
    For each canvas pixel covered by multiple images, pick the source with the
    highest local contrast (Laplacian variance of a small neighbourhood).
    Avoids motion-blur artefacts from low-quality frames dominating the mosaic.
    Falls back to feather blend when only one image covers a pixel.

last_painter_wins
    Simple last-write policy (deterministic, fastest).  Used by the GPS-only
    baseline.  Exposed here for completeness.

All modes share the same inverse-warp loop and bilinear sampling from
``drone_oversight.py``.  The render functions in this module receive
already-sampled per-image pixels and produce the final composited canvas.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mode: smart_blend
# ---------------------------------------------------------------------------

def smart_blend(
    accum: np.ndarray,
    weights_sum: np.ndarray,
    post_sharpen: bool = True,
) -> np.ndarray:
    """
    Convert a feather-weighted accumulator to a uint8 canvas.

    Parameters
    ----------
    accum        : (H, W, 3) float32 — sum of (rgb × weight) per pixel.
    weights_sum  : (H, W)    float32 — sum of weights per pixel.
    post_sharpen : bool             — apply a mild unsharp mask after compositing.

    Returns
    -------
    np.ndarray  (H, W, 3) uint8.
    """
    canvas = np.zeros((*weights_sum.shape, 3), dtype=np.uint8)
    painted = weights_sum > 0
    if not np.any(painted):
        return canvas

    canvas[painted] = np.clip(
        accum[painted] / weights_sum[painted, None], 0, 255
    ).astype(np.uint8)

    if post_sharpen:
        canvas = _unsharp_mask(canvas, painted)

    return canvas


# ---------------------------------------------------------------------------
# Mode: sharpest
# ---------------------------------------------------------------------------

class SharpestCompositor:
    """
    Per-pixel sharpness-based compositor.

    Accumulates contributions across all images and keeps track of the
    sharpest source seen so far for every canvas pixel.

    Usage
    -----
    >>> comp = SharpestCompositor(H, W)
    >>> for img, ... in images:
    ...     comp.update(canvas_y, canvas_x, sampled_rgb, sharpness_score)
    >>> canvas = comp.result()
    """

    def __init__(self, canvas_h: int, canvas_w: int) -> None:
        self.H = canvas_h
        self.W = canvas_w
        self._canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        self._best_score = np.full((canvas_h, canvas_w), -np.inf, dtype=np.float32)
        self._painted = np.zeros((canvas_h, canvas_w), dtype=bool)
        # Accumulator for feather fallback where no winner exists
        self._accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        self._weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    def update(
        self,
        canvas_y: np.ndarray,
        canvas_x: np.ndarray,
        sampled_rgb: np.ndarray,
        feather_w: np.ndarray,
        image_sharpness: float,
    ) -> None:
        """
        Update the compositor with one image's contribution.

        Parameters
        ----------
        canvas_y, canvas_x : int arrays, pixel coordinates on canvas.
        sampled_rgb        : (K, 3) float32 sampled pixel values.
        feather_w          : (K,) float32 feather weights (distance-from-edge).
        image_sharpness    : scalar global sharpness score for this image.
        """
        # Feather accumulation (blended fallback)
        self._accum[canvas_y, canvas_x] += sampled_rgb * feather_w[:, None]
        self._weight_sum[canvas_y, canvas_x] += feather_w

        # Sharpness race: per-pixel winner
        # Use image_sharpness * feather_w as per-pixel score so edge pixels
        # of a sharp image don't override centre pixels of a slightly softer one.
        pixel_score = image_sharpness * feather_w
        better = pixel_score > self._best_score[canvas_y, canvas_x]
        if not np.any(better):
            return
        gy = canvas_y[better]
        gx = canvas_x[better]
        self._canvas[gy, gx] = sampled_rgb[better]
        self._best_score[gy, gx] = pixel_score[better]
        self._painted[gy, gx] = True

    def result(self, post_sharpen: bool = True) -> np.ndarray:
        """Return the final composited uint8 canvas."""
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        painted = self._painted

        if np.any(painted):
            canvas[painted] = np.clip(self._canvas[painted], 0, 255).astype(np.uint8)

        # Fill unpainted pixels from feather fallback
        unpainted_but_covered = (~painted) & (self._weight_sum > 0)
        if np.any(unpainted_but_covered):
            canvas[unpainted_but_covered] = np.clip(
                self._accum[unpainted_but_covered]
                / self._weight_sum[unpainted_but_covered, None],
                0, 255,
            ).astype(np.uint8)

        if post_sharpen:
            all_painted = painted | unpainted_but_covered
            canvas = _unsharp_mask(canvas, all_painted)

        return canvas


def compute_image_sharpness(img: np.ndarray) -> float:
    """
    Compute a global sharpness score for an image as the variance of the
    Laplacian (Laplacian variance measure — higher = sharper).

    Parameters
    ----------
    img : (H, W, 3) uint8

    Returns
    -------
    float  Laplacian variance (higher is sharper).
    """
    if img.ndim == 3:
        # Convert to grayscale (luminance) without cv2
        gray = (
            0.299 * img[:, :, 0].astype(np.float32)
            + 0.587 * img[:, :, 1].astype(np.float32)
            + 0.114 * img[:, :, 2].astype(np.float32)
        )
    else:
        gray = img.astype(np.float32)

    # Simple 3×3 Laplacian kernel (central difference approximation)
    pad = np.pad(gray, 1, mode="edge")
    lap = (
        4.0 * pad[1:-1, 1:-1]
        - pad[:-2, 1:-1]
        - pad[2:, 1:-1]
        - pad[1:-1, :-2]
        - pad[1:-1, 2:]
    )
    return float(np.var(lap))


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _unsharp_mask(
    canvas: np.ndarray,
    painted_mask: np.ndarray,
    band_rows: int = 256,
) -> np.ndarray:
    """
    Apply a mild unsharp mask to recover detail softened by bilinear warp.

    Operates only on painted pixels (leaves black borders untouched).
    Kernel: weighted 3×3 Gaussian blur subtracted from original.

    The canvas is processed in bands of ``band_rows`` rows (with a one-row halo),
    so float temporaries stay small however large the canvas is. The result is
    identical to filtering the whole canvas at once.
    """
    if not np.any(painted_mask):
        return canvas

    height = canvas.shape[0]
    result = canvas.copy()
    for r0 in range(0, height, band_rows):
        r1 = min(r0 + band_rows, height)
        if not painted_mask[r0:r1].any():
            continue
        lo, hi = max(r0 - 1, 0), min(r1 + 1, height)       # halo rows
        band = canvas[lo:hi].astype(np.float32)
        padded = np.pad(
            band,
            ((1 if lo == r0 else 0, 1 if hi == r1 else 0), (1, 1), (0, 0)),
            mode="edge",
        )
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
        own = slice(r0 - lo, r1 - lo)
        sharpened = np.clip(band[own] * 1.22 - blurred * 0.22, 0, 255).astype(np.uint8)
        mask = painted_mask[r0:r1]
        result[r0:r1][mask] = sharpened[mask]
    return result
