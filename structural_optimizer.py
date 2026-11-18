"""
structural_optimizer.py  —  Global pose graph optimization for Task 3
======================================================================
Refines a set of per-image canvas homographies to minimise accumulated
drift using GPS soft anchors and visual pairwise constraints.

Problem
-------
After pairwise ORB refinement, images may have drifted from their GPS
positions because small per-pair errors compound across the chain.  A
global optimizer redistributes this drift by treating GPS homographies as
"soft anchors" and visual transforms as "observations".

Improvements over the previous version
---------------------------------------
1.  **Non-sequential visual edges**  (new)
    The ``optimize_poses`` function now accepts an optional ``visual_edges``
    parameter — a list of ``{"i": int, "j": int, "delta_t": [dx,dy],
    "confidence": float}`` dicts produced by ``alignment_refiner``.
    These add non-sequential pairwise constraints to the pose graph, so
    cross-strip overlaps and re-visit observations are fully exploited.

2.  **Confidence-weighted edges**  (new)
    Each visual edge contributes to the linear system with weight
    ``orb_weight × edge["confidence"]``, so high-inlier, low-deviation
    pairs pull harder than weak matches.

3.  **Per-image diagnostics**  (new)
    ``meta`` now includes:
      - ``gps_anchor_error_px``  float — max over all images of the
        L2 translation shift from GPS anchor (in normalised units × 1000).
      - ``refinement_shift_px``  float — max shift from input ``h_refined``.
      - ``visual_edges_used``    int   — number of non-sequential edges used.

4.  **Sequential chain kept as baseline**  (unchanged)
    When no ``visual_edges`` are provided (or the list is empty), the
    behaviour is identical to the previous version: only sequential chain
    constraints are added.

Backward compatibility
----------------------
``optimize_poses(h_gps, h_refined)`` with no extra args returns the same
shape and the same meta keys as before:
    ``method``, ``drift_px``, ``correction_applied``.
New meta keys are additive.

Implementation
--------------
Solver hierarchy (same priority as before):
    1.  scipy sparse least-squares   → method = "scipy_lstsq"
    2.  linear drift correction       → method = "drift_linear"
    3.  pass-through                  → method = "passthrough"
"""

from __future__ import annotations

import numpy as np

DEFAULT_GPS_WEIGHT: float = 0.4
DEFAULT_ORB_WEIGHT: float = 1.0


def optimize_poses(
    h_gps: list[np.ndarray],
    h_refined: list[np.ndarray],
    gps_weight: float = DEFAULT_GPS_WEIGHT,
    orb_weight: float = DEFAULT_ORB_WEIGHT,
    visual_edges: list[dict] | None = None,
) -> tuple[list[np.ndarray], dict]:
    """
    Globally optimize per-image canvas homographies.

    Parameters
    ----------
    h_gps        : GPS-derived (3, 3) homographies (soft anchors).
    h_refined    : ORB/visual refined (3, 3) homographies (starting point).
    gps_weight   : Weight of GPS anchor constraints.
    orb_weight   : Base weight of pairwise visual constraints.
    visual_edges : Optional list of non-sequential pairwise constraints from
                   alignment_refiner.  Each entry is a dict with keys:
                     "i"          int   — anchor image index
                     "j"          int   — constrained image index
                     "delta_t"    [dx, dy]  in canvas-normalised units
                     "confidence" float  in [0, 1]
                   All other keys (e.g. "H_cand_j") are ignored.

    Returns
    -------
    (h_opt, meta)

    h_opt : list[np.ndarray] — optimized (3, 3) homographies.
    meta  : dict with keys:
              method               str   — solver used
              drift_px             float — end-to-end drift before correction
              correction_applied   bool
              gps_anchor_error_px  float — max per-image GPS deviation (×1000)
              refinement_shift_px  float — max per-image shift from h_refined
              visual_edges_used    int   — non-sequential edges used
    """
    N = len(h_gps)
    assert N == len(h_refined), "h_gps and h_refined must have the same length."

    _base_meta = {
        "gps_anchor_error_px": 0.0,
        "refinement_shift_px": 0.0,
        "visual_edges_used": 0,
    }

    if N <= 2:
        return list(h_refined), {
            "method": "passthrough",
            "drift_px": 0.0,
            "correction_applied": False,
            **_base_meta,
        }

    # Extract 2D translations
    t_gps = np.array([H[:2, 2] for H in h_gps], dtype=np.float64)     # (N, 2)
    t_ref = np.array([H[:2, 2] for H in h_refined], dtype=np.float64)  # (N, 2)

    # Estimate accumulated drift (last image vs its GPS anchor)
    drift = float(np.linalg.norm(t_ref[-1] - t_gps[-1]))

    # Sanitize and validate visual_edges
    valid_edges: list[dict] = []
    if visual_edges:
        for e in visual_edges:
            try:
                ei, ej = int(e["i"]), int(e["j"])
                dt = e["delta_t"]
                if not (0 <= ei < N and 0 <= ej < N and ei != ej and len(dt) == 2):
                    continue
                conf = float(e.get("confidence", 1.0))
                valid_edges.append({
                    "i": ei, "j": ej,
                    "delta_t": [float(dt[0]), float(dt[1])],
                    "confidence": min(max(conf, 0.0), 1.0),
                })
            except (KeyError, TypeError, ValueError):
                continue

    # ── Solver path 1: scipy least-squares ───────────────────────────────────
    try:
        t_opt = _scipy_lstsq_graph(
            t_gps, t_ref, gps_weight, orb_weight, valid_edges
        )
        method = "scipy_lstsq"
    except Exception:
        # ── Solver path 2: linear drift correction ────────────────────────────
        try:
            t_opt = _linear_drift_correct(t_gps, t_ref)
            method = "drift_linear"
        except Exception:
            return list(h_refined), {
                "method": "passthrough",
                "drift_px": drift,
                "correction_applied": False,
                **_base_meta,
            }

    # Build output: rotation/scale from h_refined, translations from t_opt
    h_opt: list[np.ndarray] = []
    for i, H_ref in enumerate(h_refined):
        H_new = H_ref.copy()
        H_new[0, 2] = t_opt[i, 0]
        H_new[1, 2] = t_opt[i, 1]
        h_opt.append(H_new)

    applied = float(np.max(np.abs(t_opt - t_ref))) > 1e-6

    # Per-image diagnostics (scaled to "1000px reference canvas")
    gps_err = float(np.max(np.linalg.norm(t_opt - t_gps, axis=1))) * 1000.0
    ref_shift = float(np.max(np.linalg.norm(t_opt - t_ref, axis=1))) * 1000.0

    return h_opt, {
        "method": method,
        "drift_px": drift,
        "correction_applied": applied,
        "gps_anchor_error_px": round(gps_err, 2),
        "refinement_shift_px": round(ref_shift, 2),
        "visual_edges_used": len(valid_edges),
    }


# ---------------------------------------------------------------------------
# Internal solvers
# ---------------------------------------------------------------------------

def _scipy_lstsq_graph(
    t_gps: np.ndarray,
    t_ref: np.ndarray,
    gps_weight: float,
    orb_weight: float,
    visual_edges: list[dict],
) -> np.ndarray:
    """
    Sparse least-squares pose graph solver.

    Constraints (all weighted, solved per axis independently):

      GPS anchors (one per image):
        t_opt[i] ≈ t_gps[i]                          weight: gps_weight

      Sequential chain (N-1 constraints, always present for stability):
        t_opt[i] - t_opt[i-1] ≈ t_ref[i] - t_ref[i-1]  weight: orb_weight

      Non-sequential visual edges (from alignment_refiner, optional):
        t_opt[j] - t_opt[i] ≈ delta_t_ij             weight: orb_weight × confidence

    Solved independently for x and y coordinates.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr

    N = len(t_gps)
    n_gps = N
    n_seq = N - 1
    n_vis = len(visual_edges)
    total_rows = n_gps + n_seq + n_vis

    t_opt = np.zeros((N, 2), dtype=np.float64)

    for axis in range(2):
        A = lil_matrix((total_rows, N), dtype=np.float64)
        b = np.zeros(total_rows, dtype=np.float64)

        # GPS anchor rows
        for k in range(N):
            A[k, k] = gps_weight
            b[k] = gps_weight * t_gps[k, axis]

        # Sequential chain rows
        for k in range(1, N):
            row = n_gps + (k - 1)
            delta = t_ref[k, axis] - t_ref[k - 1, axis]
            A[row, k - 1] = -orb_weight
            A[row, k] = orb_weight
            b[row] = orb_weight * delta

        # Non-sequential visual edge rows
        for idx, edge in enumerate(visual_edges):
            row = n_gps + n_seq + idx
            ei = edge["i"]
            ej = edge["j"]
            dt = edge["delta_t"][axis]
            w = orb_weight * edge["confidence"]
            A[row, ei] = -w
            A[row, ej] = w
            b[row] = w * dt

        result = lsqr(A.tocsr(), b)
        t_opt[:, axis] = result[0]

    return t_opt


def _linear_drift_correct(
    t_gps: np.ndarray,
    t_ref: np.ndarray,
) -> np.ndarray:
    """
    Subtract a linearly interpolated drift from the refined translations.

    Assumes drift accumulates uniformly across the chain.  Correction for
    image i:  t_ref[i] - (i / (N-1)) × (t_ref[-1] - t_gps[-1])
    """
    N = len(t_ref)
    total_drift = t_ref[-1] - t_gps[-1]
    indices = np.arange(N, dtype=np.float64) / max(N - 1, 1)
    correction = indices[:, None] * total_drift[None, :]
    return t_ref - correction
