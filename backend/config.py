from __future__ import annotations

from exif_loader import SEKI_AGL_M
from models import CameraConfig
from models import GimbalMode

# Resource limits (protect memory on large inputs)
MAX_CSV_UPLOAD_BYTES = 50 * 1024 * 1024        # CSV / SRT telemetry
MAX_IMAGE_UPLOAD_FILES = 2000
MAX_IMAGE_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # total, all images in one upload
PREVIEW_CACHE_MAX_BYTES = 256 * 1024 * 1024     # mosaic PNG cache
PREVIEW_CACHE_MAX_ENTRIES = 32
MAX_HEATMAP_CELLS = 250_000
MAX_PLAN_WAYPOINTS = 50_000                     # mission planner output size
from pathlib import Path
import sys


def get_base_path() -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return Path(sys._MEIPASS)
    except Exception:
        return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_path()


# Serve static files (dist/)
DIST_DIR = BASE_DIR / "dist"


DEFAULT_CSV = BASE_DIR / "Drone_Forensics_Dataset.csv"


TASK3_UPLOADS_DIR = BASE_DIR / "task3_data" / "uploads"


TASK4_ORTHO_TIF = BASE_DIR / "task4_data" / "orthomosaic" / (
    "SEKI_SouthFork_DebrisFlow_2024-04-30_orthomosaic_3cm.tif"
)


TASK4_ORTHO_XML = BASE_DIR / "task4_data" / "orthomosaic" / (
    "SEKI_SouthFork_DebrisFlow_2024-04-30_orthomosaic_metadata.xml"
)


TASK4_UPLOAD_DIR = BASE_DIR / "task4_data" / "uploads"


TASK4_UPLOAD_TIF = TASK4_UPLOAD_DIR / "task4_reference_upload.tif"


TASK4_UPLOAD_JPG = TASK4_UPLOAD_DIR / "task4_reference_upload.jpg"


# Default camera config (90°×60° FOV, nadir)
_DEFAULT_CAMERA = CameraConfig(fov_h=90.0, fov_v=60.0, gimbal_mode=GimbalMode.NADIR)


_DEFAULT_IMAGE_MISSION_DIRS: dict[str, tuple[str, float]] = {
    "SEKI_SOUTHFORK": ("task3_data/seki_southfork/images", SEKI_AGL_M),
    "CD_4TH_AVE": ("task3_data/cd_4th_ave_subset/images", 110.0),
}
