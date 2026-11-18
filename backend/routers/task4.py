from __future__ import annotations

from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import Response
from pathlib import Path
import io
import shutil
import tempfile
from backend.config import TASK4_UPLOAD_DIR, TASK4_UPLOAD_JPG, TASK4_UPLOAD_TIF
from backend.services.task4 import _get_task4_reference_metadata, _get_task4_reference_tif_path, _load_task4_geotiff_metadata, _load_task4_jpg_metadata, _task4_reference_available

router = APIRouter()


@router.get("/task4/reference")
async def get_task4_reference_metadata():
    return _get_task4_reference_metadata()


@router.post("/task4/reference/upload")
async def upload_task4_reference(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename.endswith((".tif", ".tiff", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Supported files: GeoTIFF (.tif/.tiff) or JPG with GPS EXIF (.jpg/.jpeg).")

    TASK4_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    is_tif = filename.endswith((".tif", ".tiff"))
    target_path = TASK4_UPLOAD_TIF if is_tif else TASK4_UPLOAD_JPG
    other_path = TASK4_UPLOAD_JPG if is_tif else TASK4_UPLOAD_TIF
    with tempfile.TemporaryDirectory(prefix=".reference_", dir=TASK4_UPLOAD_DIR) as staging:
        staged_path = Path(staging) / target_path.name
        try:
            with staged_path.open("wb") as out_fh:
                shutil.copyfileobj(file.file, out_fh)
            loader = _load_task4_geotiff_metadata if is_tif else _load_task4_jpg_metadata
            metadata = loader(staged_path)
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        staged_path.replace(target_path)
        other_path.unlink(missing_ok=True)
    return {**metadata, "source": "uploaded", "filename": target_path.name}


@router.get("/task4/reference/preview")
async def get_task4_reference_preview(scale: float = 0.02):
    if not _task4_reference_available():
        raise HTTPException(status_code=404, detail="Task 4 reference data not found.")

    scale = max(0.005, min(0.05, scale))

    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")

    PILImage.MAX_IMAGE_PIXELS = None
    with PILImage.open(_get_task4_reference_tif_path()) as img:
        img = img.convert("RGB")
        new_w = max(1, round(img.width * scale))
        new_h = max(1, round(img.height * scale))
        img = img.resize((new_w, new_h), PILImage.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
