"""
models.py
=========
מודלים משותפים למערכת Drone Oversight.
כל המשימות (Tasks 1-6) מייבאות מכאן כדי להבטיח עקביות בסוגי הנתונים.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# מודל קואורדינטת GPS משותף
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPSCoordinate:
    """
    נקודת GPS בלתי-ניתנת לשינוי, מבוטאת בדרגות עשרוניות.
    frozen=True אומר שלא ניתן לשנות אובייקט לאחר יצירתו — בטוח יותר לחישובים.
    """
    lat: float   # קו רוחב  (-90 עד +90)
    lon: float   # קו אורך  (-180 עד +180)
    alt: float = 0.0  # גובה מעל הקרקע במטרים

    def __repr__(self) -> str:
        # מחזיר מחרוזת קריאה לצורך הדפסה
        return f"GPSCoordinate(lat={self.lat:.6f}, lon={self.lon:.6f}, alt={self.alt:.1f}m)"

    def to_array(self) -> np.ndarray:
        # ממיר את הנקודה למערך numpy — שימושי לחשבון וקטורי ב-numpy
        return np.array([self.lat, self.lon, self.alt], dtype=np.float64)


# ---------------------------------------------------------------------------
# ספירות (Enums) — ערכים מוגדרים מראש
# ---------------------------------------------------------------------------

class GimbalMode(Enum):
    """מצב כיוון המצלמה."""
    NADIR   = auto()   # מצלמה מכוונת ישר למטה — Task 1 ממומש
    OBLIQUE = auto()   # מצלמה מוטה בזווית — עתידי (stub)


class CoverageStatus(Enum):
    """מצב כיסוי של תא בגריד — משמש ב-Task 2 (Heatmap)."""
    UNCOVERED = auto()  # לא צולם כלל
    PARTIAL   = auto()  # צולם חלקית
    COVERED   = auto()  # צולם לחלוטין


# ---------------------------------------------------------------------------
# קונפיגורציית מצלמה
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """
    מגדיר את המצלמה הפיזית ואת חיבורה לרחפן (gimbal).
    משותף לכל המשימות שמשתמשות בחישוב שדה ראייה.
    """
    fov_h: float                          # שדה ראייה אופקי (horizontal FOV) בדרגות
    fov_v: Optional[float] = None         # שדה ראייה אנכי — אם None, יהיה זהה לאופקי
    gimbal_mode: GimbalMode = GimbalMode.NADIR  # מצב ה-gimbal (ברירת מחדל: ישר למטה)
    gimbal_pitch: float = 0.0            # זווית הטיה של ה-gimbal בדרגות (רלוונטי רק ל-OBLIQUE)

    def __post_init__(self) -> None:
        # אם לא הוגדר FOV אנכי — משתמשים באופקי (מצלמה ריבועית / סימטרית)
        if self.fov_v is None:
            self.fov_v = self.fov_h


# ---------------------------------------------------------------------------
# מצב טלמטריה של הרחפן (snapshot)
# ---------------------------------------------------------------------------

@dataclass
class DroneTelemetry:
    """
    "צילום מצב" אחד של הרחפן בזמן נתון — כולל מיקום, כיוון, מצלמה וחותמת זמן.
    Task 1 מקבל אובייקט כזה ומחשב את טביעת הרגל על הקרקע.
    """
    position: GPSCoordinate   # מיקום GPS נוכחי (כולל גובה AGL)
    heading: float             # כיוון הרחפן בדרגות מהצפון (0=צפון, 90=מזרח)
    camera: CameraConfig       # קונפיגורציית המצלמה ברגע הצילום
    timestamp: float = 0.0     # חותמת זמן Unix — 0.0 אם לא ידוע


# ---------------------------------------------------------------------------
# Task 3 Orthomosaic Pipeline — Dataclasses for DJI image stitching
# ---------------------------------------------------------------------------

@dataclass
class DJIImageMetadata:
    """DJI image metadata from EXIF + XMP."""
    path: str
    lat: float
    lon: float
    alt_abs: float  # WGS84 ellipsoid altitude (metres)
    alt_rel: float  # Relative altitude above ground (metres)
    gimbal_yaw: float     # Rotation around Z-axis (degrees)
    gimbal_pitch: float   # Tilt angle (degrees, typically −90 for nadir)
    gimbal_roll: float    # Roll angle (degrees)
    flight_yaw: float     # Flight heading (degrees from North)
    focal_length_mm: float
    image_width_px: int
    image_height_px: int
    sensor_width_mm: float = 17.3   # DJI Mavic 3 default
    sensor_height_mm: float = 13.0


@dataclass
class MatchResult:
    """Result of feature matching between two images."""
    i: int                        # Anchor image index
    j: int                        # Matched image index
    src_pts: np.ndarray          # Source keypoints (Nx2)
    dst_pts: np.ndarray          # Destination keypoints (Nx2)
    inlier_mask: np.ndarray      # Boolean mask of inliers (N,)
    confidence: float            # Match confidence [0, 1]
    n_inliers: int              # Number of inliers
    mean_reproj_err: float      # Mean reprojection error (pixels)


@dataclass
class PairTransform:
    """Homography transformation between two images."""
    i: int                # Anchor image index
    j: int                # Matched image index
    H: np.ndarray         # 3×3 homography matrix
    confidence: float     # Transformation confidence [0, 1]


@dataclass
class GlobalPose:
    """Optimized 6-DOF pose for a single image in canvas space."""
    image_idx: int           # Image index
    H_canvas: np.ndarray    # 3×3 homography mapping image pixels → canvas pixels
    tx: float               # X translation
    ty: float               # Y translation
    theta: float            # Rotation angle (radians)
    sx: float               # X scale
    sy: float               # Y scale


@dataclass
class OrthoMosaicResult:
    """Final orthomosaic output."""
    canvas: np.ndarray         # H×W×3 uint8 RGB image
    seam_mask: np.ndarray      # H×W int32, value = source image index
    poses: list[GlobalPose]
    crs_wkt: str              # Coordinate reference system (WKT string)
    affine_transform: tuple   # (pixel_size_x, 0, origin_x, 0, pixel_size_y, origin_y)
    diagnostics: dict         # Metadata: coverage, gsd, etc.
