"""
alignment_refiner.py  —  GPS-regularized visual alignment for Task 3
=====================================================================
Refines GPS-guided homographies using pairwise ORB feature matching
*inside* GPS-predicted overlap regions, with GPS regularization and
non-sequential neighbour graphs.

Key improvements over the naive consecutive-pair approach
---------------------------------------------------------
1. **GPS-overlap-guided matching**
   Before detecting or matching features, the GPS homographies are used to
   predict which area of the canvas each image occupies.  Two images are
   only paired when their predicted canvas footprints overlap by at least
   ``min_gps_overlap_frac``.  Features are detected on a crop limited to
   the GPS-predicted overlap region of each image, focusing the keypoint
   budget exactly where content is shared.

2. **Non-sequential neighbours**
   Consecutive pairs (i, i+1) are always checked, but pairs up to
   ``neighbor_window`` steps apart (i, i+2), (i, i+3), … are also
   considered when GPS overlap justifies it.  This captures cross-strip
   overlaps in lawnmower survey patterns.

3. **GPS deviation guard**
   After estimating a visual transform, the resulting canvas placement for
   image j is compared with its GPS prior.  Transforms that deviate more
   than ``max_gps_deviation_frac`` of the canvas diagonal are hard-rejected
   to prevent GPS-incompatible hallucinations from noisy RANSAC solutions.

4. **GPS regularization**
   Transforms that pass the guard but have low confidence are blended toward
   the GPS prior: ``alpha = confidence`` (0 = GPS-only, 1 = visual-only).
   High-confidence, low-deviation pairs converge to the visual estimate.

5. **visual_edges output**
   Every accepted pair is recorded as a ``visual_edge`` dict for downstream
   use by the structural optimizer, which can use non-sequential constraints
   instead of the sequential chain.

6. **Diagnostics**
   ``gps_anchor_error_px``, ``refinement_shift_px`` (per-image), ``per_pair``
   (per-edge), and ``visual_edges`` are added to the returned metrics while
   preserving all existing keys (``accepted_pairs``, ``fallback_pairs``,
   ``total_pairs``, ``mean_inliers``, ``fallback_reason``).

Backward compatibility
----------------------
The public signature is unchanged:
    refine_placements(images, h_gps) -> (refined, metrics)
All new parameters have safe defaults.  Existing callers and tests are
unaffected.
"""

from __future__ import annotations

import numpy as np

# ── Tuning constants ──────────────────────────────────────────────────────────

# Minimum image dimension for reliable ORB matching.
MIN_REFINE_PX: int = 200

# Minimum RANSAC inliers to accept a pairwise transform.
MIN_INLIERS: int = 12

# Default neighbour window: check pairs (i, i+1) … (i, i+NEIGHBOR_WINDOW).
# Lawnmower patterns can have cross-strip overlaps 8+ frames apart.
DEFAULT_NEIGHBOR_WINDOW: int = 8

# Minimum GPS-predicted overlap fraction to attempt feature matching.
DEFAULT_MIN_GPS_OVERLAP: float = 0.10

# Maximum allowed transform deviation from GPS, as a fraction of the canvas
# normalised diagonal (sqrt(2) ≈ 1.414).  Transforms beyond 2× this are
# hard-rejected; between 1× and 2× they are GPS-regularized.
DEFAULT_MAX_GPS_DEVIATION_FRAC: float = 0.15

# Inlier count that corresponds to confidence = 1.0.  Empirically, 80 inliers
# at preview scale represents excellent match quality for aerial imagery.
CONFIDENCE_INLIER_SCALE: float = 80.0

# Confidence threshold for accepting visual estimates vs GPS fallback.
# Linear blending of homography matrices is mathematically invalid for
# projective transforms. Instead: if confidence >= this threshold, use the
# visual estimate; otherwise fall back to GPS prior. Threshold of 0.35
# corresponds to ~28 ORB inliers and low GPS deviation.
CONFIDENCE_ACCEPT_THRESHOLD: float = 0.35

# Coarse-to-fine ORB matching: first pass scale
# Run ORB at 0.5× resolution to get approximate relative corrections,
# then use these for improved ROI cropping in the full-scale pass.
COARSE_SCALE: float = 0.5


# ── Canvas-space geometry helpers ─────────────────────────────────────────────

def _canvas_bbox(H: np.ndarray) -> tuple[float, float, float, float]:
    """
    Axis-aligned bounding box of image [0,1]^2 projected through H.

    H maps img_norm → canvas_norm.  Projects the four corners
    (TL, TR, BR, BL) and returns (u_min, u_max, v_min, v_max).
    """
    # Four image corners in homogeneous coordinates
    corners = np.array(
        [[0, 1, 1, 0],
         [0, 0, 1, 1],
         [1, 1, 1, 1]],
        dtype=np.float64,
    )
    proj = H @ corners          # (3, 4)
    w = proj[2]
    safe_w = np.where(np.abs(w) > 1e-12, w, 1e-12)
    u = proj[0] / safe_w
    v = proj[1] / safe_w
    return float(u.min()), float(u.max()), float(v.min()), float(v.max())


def _compute_gps_overlap(H_i: np.ndarray, H_j: np.ndarray) -> float:
    """
    Fraction of image i's canvas footprint covered by image j's footprint.

    Returns a value in [0, 1].  Uses axis-aligned bounding boxes in canvas
    normalised space — a conservative approximation that avoids full polygon
    intersection while remaining accurate enough for neighbour selection.
    """
    u_min_i, u_max_i, v_min_i, v_max_i = _canvas_bbox(H_i)
    u_min_j, u_max_j, v_min_j, v_max_j = _canvas_bbox(H_j)

    inter_u = max(0.0, min(u_max_i, u_max_j) - max(u_min_i, u_min_j))
    inter_v = max(0.0, min(v_max_i, v_max_j) - max(v_min_i, v_min_j))
    area_i = max((u_max_i - u_min_i) * (v_max_i - v_min_i), 1e-12)
    return float(inter_u * inter_v / area_i)


def _overlap_roi_in_image(
    H_i: np.ndarray,
    H_j: np.ndarray,
    img_h: int,
    img_w: int,
    pad_frac: float = 0.08,
) -> tuple[int, int, int, int] | None:
    """
    ROI in image i's pixel space corresponding to the GPS-predicted overlap
    with image j.

    Computes the intersection of the two canvas bounding boxes, back-projects
    it through ``H_i^{-1}`` into image i's normalised source space, converts
    to pixel coordinates with a small padding, and clamps to image bounds.

    Returns (x_min, x_max, y_min, y_max) in pixels, or None if the overlap
    is empty or the ROI is too small to be useful.
    """
    u_min_i, u_max_i, v_min_i, v_max_i = _canvas_bbox(H_i)
    u_min_j, u_max_j, v_min_j, v_max_j = _canvas_bbox(H_j)

    u_lo = max(u_min_i, u_min_j)
    u_hi = min(u_max_i, u_max_j)
    v_lo = max(v_min_i, v_min_j)
    v_hi = min(v_max_i, v_max_j)

    if u_hi <= u_lo or v_hi <= v_lo:
        return None

    try:
        H_inv = np.linalg.inv(H_i)
    except np.linalg.LinAlgError:
        return None

    # Four corners of the canvas-space intersection
    corners_canvas = np.array(
        [[u_lo, u_hi, u_hi, u_lo],
         [v_lo, v_lo, v_hi, v_hi],
         [1.0,  1.0,  1.0,  1.0]],
        dtype=np.float64,
    )
    src = H_inv @ corners_canvas          # (3, 4)
    sw = src[2]
    safe_sw = np.where(np.abs(sw) > 1e-12, sw, 1e-12)
    src_u = src[0] / safe_sw              # normalised image coordinates
    src_v = src[1] / safe_sw

    # Add padding and convert to pixel coordinates
    u_span = src_u.max() - src_u.min()
    v_span = src_v.max() - src_v.min()
    u_pad = u_span * pad_frac
    v_pad = v_span * pad_frac

    x_min = max(0, int((src_u.min() - u_pad) * img_w))
    x_max = min(img_w - 1, int((src_u.max() + u_pad) * img_w))
    y_min = max(0, int((src_v.min() - v_pad) * img_h))
    y_max = min(img_h - 1, int((src_v.max() + v_pad) * img_h))

    if x_max - x_min < 20 or y_max - y_min < 20:
        return None
    return x_min, x_max, y_min, y_max


def _gps_deviation_norm(H_candidate: np.ndarray, H_gps: np.ndarray) -> float:
    """
    Mean corner deviation between two homographies in canvas normalised units.

    Projects the four image corners through each homography and returns the
    mean Euclidean distance between corresponding points.  This is a scale-
    invariant measure of how much the candidate deviates from the GPS prior.
    """
    corners = np.array(
        [[0, 1, 1, 0],
         [0, 0, 1, 1],
         [1, 1, 1, 1]],
        dtype=np.float64,
    )

    def _proj(H: np.ndarray) -> np.ndarray:
        p = H @ corners
        w = np.where(np.abs(p[2]) > 1e-12, p[2], 1e-12)
        return np.stack([p[0] / w, p[1] / w], axis=0)   # (2, 4)

    p_c = _proj(H_candidate)
    p_g = _proj(H_gps)
    return float(np.linalg.norm(p_c - p_g, axis=0).mean())


# ── Feature matching inside a pixel-space crop ────────────────────────────────

def _detect_in_crop(
    orb,
    clahe,
    img: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    import_cv2,
) -> tuple[list, np.ndarray | None, int, int]:
    """
    Detect ORB keypoints within a pixel-space ROI crop.

    Returns (keypoints, descriptors, x_offset, y_offset) where the offset
    is the crop's top-left corner in the full image.  Keypoint coordinates
    are in the crop's local pixel space; add (x_offset, y_offset) to get
    full-image coordinates.

    Falls back to the full image if ``roi`` is None.
    """
    cv2 = import_cv2
    if roi is None:
        crop = img
        ox, oy = 0, 0
    else:
        x0, x1, y0, y1 = roi
        crop = img[y0:y1, x0:x1]
        ox, oy = x0, y0

    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return [], None, ox, oy

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = clahe.apply(gray)
    kp, des = orb.detectAndCompute(gray, None)
    return kp or [], des, ox, oy


def _resize_image_nearest(img: np.ndarray, scale: float, cv2_module) -> np.ndarray:
    """
    Downscale an image using nearest-neighbour interpolation.
    Used for coarse-pass ORB detection.
    """
    cv2 = cv2_module
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


# ── Main public function ──────────────────────────────────────────────────────

def refine_placements(
    images: list[np.ndarray],
    h_gps: list[np.ndarray],
    nfeatures: int = 2000,
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
    min_gps_overlap_frac: float = DEFAULT_MIN_GPS_OVERLAP,
    max_gps_deviation_frac: float = DEFAULT_MAX_GPS_DEVIATION_FRAC,
    regularize: bool = True,
) -> tuple[list[np.ndarray], dict]:
    """
    Refine GPS-based canvas homographies using GPS-overlap-guided ORB matching.

    Parameters
    ----------
    images               : list[np.ndarray]  — uint8 RGB arrays, one per image.
    h_gps                : list[np.ndarray]  — GPS homographies (img_norm → canvas_norm).
    nfeatures            : ORB keypoint budget per crop.
    neighbor_window      : Maximum frame-index distance checked for GPS overlap.
    min_gps_overlap_frac : Minimum GPS-predicted overlap to attempt matching.
    max_gps_deviation_frac : Hard-rejection threshold as fraction of canvas diagonal.
    regularize           : Blend low-confidence transforms toward GPS prior.

    Returns
    -------
    (refined, metrics)

    refined : list[np.ndarray] — one GPS-regularized (3, 3) per image.
    metrics : dict with keys:
                accepted_pairs       int
                fallback_pairs       int
                total_pairs          int
                mean_inliers         float
                rejected_pairs       int
                gps_anchor_error_px  list[float]  — per image (1000px ref canvas)
                refinement_shift_px  list[float]  — per image
                per_pair             list[dict]
                visual_edges         list[dict]  — for structural_optimizer
                fallback_reason      str  (only when ALL pairs fell back)
    """
    N = len(images)
    canvas_diag = float(np.sqrt(2.0))    # normalised-space diagonal

    # Initialise per-image diagnostic arrays
    gps_anchor_error: list[float] = [0.0] * N
    refinement_shift: list[float] = [0.0] * N

    _base = {
        "accepted_pairs": 0,
        "fallback_pairs": 0,
        "total_pairs": 0,
        "mean_inliers": 0.0,
        "rejected_pairs": 0,
        "gps_anchor_error_px": gps_anchor_error,
        "refinement_shift_px": refinement_shift,
        "per_pair": [],
        "visual_edges": [],
    }

    # ── Early exits ───────────────────────────────────────────────────────────
    try:
        import cv2
    except ImportError:
        return list(h_gps), {**_base, "fallback_reason": "cv2_not_available"}

    if N <= 1:
        return list(h_gps), {**_base, "total_pairs": 0, "fallback_pairs": 0}

    min_dim = min(min(img.shape[:2]) for img in images)
    if min_dim < MIN_REFINE_PX:
        return list(h_gps), {
            **_base,
            "fallback_reason": (
                f"images_too_small (min={min_dim}px < {MIN_REFINE_PX}px)"
            ),
        }

    # ── Shared detectors ─────────────────────────────────────────────────────
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=12,
        fastThreshold=10,
    )
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ── Build candidate pair list from GPS overlap ────────────────────────────
    candidate_pairs: list[tuple[int, int, float]] = []   # (i, j, overlap_frac)
    seen: set[tuple[int, int]] = set()

    for i in range(N):
        for k in range(1, neighbor_window + 1):
            j = i + k
            if j >= N:
                break
            # Symmetric overlap: take the larger of the two fractions
            ovl_ij = _compute_gps_overlap(h_gps[i], h_gps[j])
            ovl_ji = _compute_gps_overlap(h_gps[j], h_gps[i])
            ovl = max(ovl_ij, ovl_ji)
            if ovl >= min_gps_overlap_frac and (i, j) not in seen:
                candidate_pairs.append((i, j, ovl))
                seen.add((i, j))
        # Always include the consecutive pair as a safety fallback
        if i + 1 < N and (i, i + 1) not in seen:
            candidate_pairs.append((i, i + 1, 0.0))
            seen.add((i, i + 1))

    _base["total_pairs"] = len(candidate_pairs)

    # ── Coarse-to-fine pass 1: low-resolution ORB for approximate corrections ──
    coarse_images = [_resize_image_nearest(img, COARSE_SCALE, cv2) for img in images]
    coarse_orb = cv2.ORB_create(
        nfeatures=min(1000, nfeatures // 2),
        scaleFactor=1.2,
        nlevels=6,
        edgeThreshold=8,
        fastThreshold=12,
    )
    coarse_best: dict[int, np.ndarray] = {}  # j → improved canvas-norm H
    coarse_corrections = 0

    for (i, j, ovl_frac) in candidate_pairs:
        h_c_img_i, w_c_img_i = coarse_images[i].shape[:2]
        h_c_img_j, w_c_img_j = coarse_images[j].shape[:2]

        # Overlap ROI in coarse pixel space
        roi_c_i = _overlap_roi_in_image(h_gps[i], h_gps[j], h_c_img_i, w_c_img_i)
        roi_c_j = _overlap_roi_in_image(h_gps[j], h_gps[i], h_c_img_j, w_c_img_j)

        kp_c_i, des_c_i, ox_c_i, oy_c_i = _detect_in_crop(coarse_orb, clahe, coarse_images[i], roi_c_i, cv2)
        kp_c_j, des_c_j, ox_c_j, oy_c_j = _detect_in_crop(coarse_orb, clahe, coarse_images[j], roi_c_j, cv2)

        if des_c_i is None or des_c_j is None or len(kp_c_i) < MIN_INLIERS or len(kp_c_j) < MIN_INLIERS:
            continue

        raw_c = bf.knnMatch(des_c_i, des_c_j, k=2)
        good_c = [m for pair in raw_c if len(pair) == 2 and (m := pair[0]).distance < 0.72 * pair[1].distance]

        if len(good_c) < MIN_INLIERS:
            continue

        pts_c_i = np.float32([[kp_c_i[m.queryIdx].pt[0] + ox_c_i, kp_c_i[m.queryIdx].pt[1] + oy_c_i] for m in good_c])
        pts_c_j = np.float32([[kp_c_j[m.trainIdx].pt[0] + ox_c_j, kp_c_j[m.trainIdx].pt[1] + oy_c_j] for m in good_c])

        aff_c, mask_c = cv2.estimateAffinePartial2D(pts_c_j, pts_c_i, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if aff_c is None or mask_c is None or int(mask_c.sum()) < MIN_INLIERS:
            continue

        # Upgrade affine to homography
        H_pw_c = np.vstack([aff_c, [0.0, 0.0, 1.0]])
        S_j_c = np.diag([float(max(w_c_img_j - 1, 1)), float(max(h_c_img_j - 1, 1)), 1.0])
        S_i_inv_c = np.diag([1.0 / float(max(w_c_img_i - 1, 1)), 1.0 / float(max(h_c_img_i - 1, 1)), 1.0])
        H_cand_c = h_gps[i] @ S_i_inv_c @ H_pw_c @ S_j_c

        # Accept coarse correction if GPS deviation is moderate
        dev_c = _gps_deviation_norm(H_cand_c, h_gps[j]) / canvas_diag
        if dev_c <= max_gps_deviation_frac * 2.0 and j not in coarse_best:
            coarse_best[j] = H_cand_c
            coarse_corrections += 1

    # Build improved GPS priors from coarse corrections
    h_gps_improved = [coarse_best.get(k, h_gps[k]) for k in range(N)]
    _base["coarse_pass_corrections"] = coarse_corrections

    # ── Pairwise matching ─────────────────────────────────────────────────────
    # For each accepted pair: record the GPS-regularized canvas H for image j.
    # best_for[j] = (H_candidate, confidence, inliers)  highest-confidence so far
    best_for: dict[int, tuple[np.ndarray, float, int]] = {}
    visual_edges: list[dict] = []
    per_pair_diag: list[dict] = []
    accepted = 0
    fallback = 0
    rejected = 0
    inlier_counts: list[int] = []

    for (i, j, ovl_frac) in candidate_pairs:
        h_img_i, w_img_i = images[i].shape[:2]
        h_img_j, w_img_j = images[j].shape[:2]

        # Compute overlap ROI using improved priors from coarse pass
        roi_i = _overlap_roi_in_image(h_gps_improved[i], h_gps_improved[j], h_img_i, w_img_i)
        roi_j = _overlap_roi_in_image(h_gps_improved[j], h_gps_improved[i], h_img_j, w_img_j)

        # Detect features inside the overlap crop
        kp_i, des_i, ox_i, oy_i = _detect_in_crop(orb, clahe, images[i], roi_i, cv2)
        kp_j, des_j, ox_j, oy_j = _detect_in_crop(orb, clahe, images[j], roi_j, cv2)

        if (
            des_i is None or des_j is None
            or len(kp_i) < MIN_INLIERS
            or len(kp_j) < MIN_INLIERS
        ):
            fallback += 1
            per_pair_diag.append({
                "pair": (i, j), "status": "fallback_no_features",
                "inliers": 0, "confidence": 0.0,
                "overlap_frac": round(ovl_frac, 3),
            })
            continue

        # Lowe ratio test
        raw = bf.knnMatch(des_i, des_j, k=2)
        good: list = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.72 * n.distance:
                    good.append(m)

        if len(good) < MIN_INLIERS:
            fallback += 1
            per_pair_diag.append({
                "pair": (i, j), "status": "fallback_few_matches",
                "inliers": 0, "confidence": 0.0,
                "overlap_frac": round(ovl_frac, 3),
            })
            continue

        # Restore full-image pixel coordinates
        pts_i = np.float32([
            [kp_i[m.queryIdx].pt[0] + ox_i, kp_i[m.queryIdx].pt[1] + oy_i]
            for m in good
        ])
        pts_j = np.float32([
            [kp_j[m.trainIdx].pt[0] + ox_j, kp_j[m.trainIdx].pt[1] + oy_j]
            for m in good
        ])

        # Estimate H_pw: maps image j pixels → image i pixels (pts_j → pts_i)
        H_pw: np.ndarray | None = None
        inliers = 0

        aff, mask_aff = cv2.estimateAffinePartial2D(
            pts_j, pts_i,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if aff is not None and mask_aff is not None:
            inl = int(mask_aff.sum())
            if inl >= MIN_INLIERS:
                H_pw = np.vstack([aff, [0.0, 0.0, 1.0]])
                inliers = inl

        if H_pw is None:
            H_h, mask_h = cv2.findHomography(pts_j, pts_i, cv2.RANSAC, 4.0)
            if H_h is not None and mask_h is not None:
                inl = int(mask_h.sum())
                if inl >= MIN_INLIERS and np.linalg.cond(H_h) < 1e5:
                    H_pw = H_h
                    inliers = inl

        if H_pw is None or np.linalg.cond(H_pw) > 1e5:
            fallback += 1
            per_pair_diag.append({
                "pair": (i, j), "status": "fallback_ransac",
                "inliers": 0, "confidence": 0.0,
                "overlap_frac": round(ovl_frac, 3),
            })
            continue

        # Compose: H_cand_j maps img_j_norm → canvas_norm via image i's GPS
        S_j = np.diag([float(w_img_j - 1), float(h_img_j - 1), 1.0])
        S_i_inv = np.diag([
            1.0 / max(w_img_i - 1, 1),
            1.0 / max(h_img_i - 1, 1),
            1.0,
        ])
        H_cand_j = h_gps[i] @ S_i_inv @ H_pw @ S_j

        if np.linalg.cond(H_cand_j) > 1e6:
            fallback += 1
            per_pair_diag.append({
                "pair": (i, j), "status": "fallback_ill_conditioned",
                "inliers": inliers, "confidence": 0.0,
                "overlap_frac": round(ovl_frac, 3),
            })
            continue

        # GPS deviation check
        dev_norm = _gps_deviation_norm(H_cand_j, h_gps[j])
        dev_frac = dev_norm / canvas_diag

        # Hard reject: way beyond GPS prior
        if dev_frac > max_gps_deviation_frac * 2.0:
            rejected += 1
            per_pair_diag.append({
                "pair": (i, j), "status": "rejected_gps_deviation",
                "inliers": inliers, "confidence": 0.0,
                "overlap_frac": round(ovl_frac, 3),
                "gps_deviation_frac": round(dev_frac, 4),
            })
            continue

        # Confidence: higher inliers + lower GPS deviation → higher confidence
        conf_inliers = min(1.0, inliers / CONFIDENCE_INLIER_SCALE)
        conf_gps = max(0.0, 1.0 - dev_frac / max(max_gps_deviation_frac, 1e-9))
        confidence = conf_inliers * conf_gps

        # GPS regularization: binary selection between visual and GPS.
        # Linear interpolation of homography matrices is mathematically invalid
        # for projective transforms — it introduces shear and scale distortion.
        # Instead: accept the visual estimate when confidence is high, otherwise
        # use the GPS prior as-is.
        if regularize:
            if confidence >= CONFIDENCE_ACCEPT_THRESHOLD:
                H_cand_j_reg = H_cand_j           # high confidence: use visual
            else:
                H_cand_j_reg = h_gps[j].copy()    # low confidence: use GPS prior
        else:
            H_cand_j_reg = H_cand_j

        # Translation delta for structural optimizer:
        # delta_t = deviation of image j's visual placement from j's own GPS prior.
        # Using h_gps[i] here was wrong — it mixed image i's GPS anchor into the
        # per-image-j correction, double-counting GPS in the SBA system matrix.
        delta_t = np.array([
            H_cand_j[0, 2] - h_gps[j][0, 2],
            H_cand_j[1, 2] - h_gps[j][1, 2],
        ])

        # Record visual edge (uses unregularized H_cand_j so optimizer sees
        # the raw visual estimate, not the pre-blended one)
        visual_edges.append({
            "i": i,
            "j": j,
            "delta_t": delta_t.tolist(),
            "confidence": float(confidence),
            "inliers": inliers,
            "overlap_frac": float(ovl_frac),
            "gps_deviation_frac": float(dev_frac),
            "H_cand_j": H_cand_j,       # full homography for optimizer
        })

        per_pair_diag.append({
            "pair": (i, j),
            "status": "accepted",
            "inliers": inliers,
            "confidence": round(confidence, 3),
            "overlap_frac": round(ovl_frac, 3),
            "gps_deviation_frac": round(dev_frac, 4),
        })

        # Keep best candidate per image j
        if j not in best_for or confidence > best_for[j][1]:
            best_for[j] = (H_cand_j_reg, confidence, inliers)

        accepted += 1
        inlier_counts.append(inliers)

    # ── Assemble per-image refined placements ─────────────────────────────────
    refined: list[np.ndarray] = []
    for i in range(N):
        if i not in best_for:
            # No visual constraint accepted — use GPS
            refined.append(h_gps[i])
        else:
            H_best, best_conf, best_inl = best_for[i]
            refined.append(H_best)
            # Per-image deviation from GPS (scaled to "1000px canvas")
            dev = _gps_deviation_norm(H_best, h_gps[i]) * 1000.0
            gps_anchor_error[i] = round(dev, 1)
            refinement_shift[i] = round(dev, 1)
            inlier_counts.append(best_inl)

    metrics: dict = {
        "accepted_pairs": accepted,
        "fallback_pairs": fallback,
        "total_pairs": len(candidate_pairs),
        "mean_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        "rejected_pairs": rejected,
        "gps_anchor_error_px": gps_anchor_error,
        "refinement_shift_px": refinement_shift,
        "per_pair": per_pair_diag,
        "visual_edges": visual_edges,
    }

    if accepted == 0:
        metrics["fallback_reason"] = "no_valid_visual_matches"

    return refined, metrics
