"""Compatibility entry point: ``uvicorn api:app``.

The application now lives in the ``backend`` package (see backend/main.py).
Names below are re-exported for tests and tooling.
"""
from backend.main import app  # noqa: F401
from backend.state import _custom_drones, _image_missions, _image_preview_cache, _missions  # noqa: F401
from backend.services.images import _detect_clusters, _prepare_image_preview_inputs  # noqa: F401
from backend.services.telemetry import _row_to_footprint, _parse_csv  # noqa: F401
