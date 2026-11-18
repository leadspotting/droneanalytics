import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

import api as api_module
from drone_oversight import MosaicResult
from models import GPSCoordinate


def _fake_row(tmp_path: Path, index: int = 0) -> dict:
    return {
        "path":            str(tmp_path / f"img_{index}.jpg"),
        "lat":             36.340 + index * 0.001,
        "lon":             -118.760 + index * 0.001,
        "agl_m":           110.0,
        "alt_ellipsoid_m": 1465.0,
        "timestamp":       "",
        "heading":         0.0,
        "width_px":        4928,
        "height_px":       3264,
        "fov_h":           65.8,
        "fov_v":           46.4,
    }


class TestMosaicExportRoute:

    def setup_method(self):
        api_module._image_missions.clear()

    def test_unknown_mission_returns_404(self):
        client = TestClient(api_module.app)
        resp = client.get("/image_mission/DOES_NOT_EXIST/mosaic/export")
        assert resp.status_code == 404

    def test_422_when_not_enough_loadable_images(self, tmp_path):
        api_module._image_missions["EXPORT_EMPTY"] = [_fake_row(tmp_path, 0)]
        client = TestClient(api_module.app)
        resp = client.get("/image_mission/EXPORT_EMPTY/mosaic/export")
        assert resp.status_code == 422

    def test_returns_zip_with_png_json_and_worldfile(self, tmp_path):
        tiny_img = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        fake_mosaic = MosaicResult(
            image=np.zeros((5, 8, 3), dtype=np.uint8),
            bounding_box=[
                GPSCoordinate(lat=36.354000, lon=-118.768000),
                GPSCoordinate(lat=36.354000, lon=-118.754000),
                GPSCoordinate(lat=36.338000, lon=-118.754000),
                GPSCoordinate(lat=36.338000, lon=-118.768000),
            ],
            gsd_m=0.579,
        )

        api_module._image_missions["EXPORT_OK"] = [_fake_row(tmp_path, i) for i in range(2)]

        with patch("PIL.Image.open", return_value=tiny_img), \
             patch("drone_oversight.ImageStitcher.stitch", return_value=fake_mosaic):
            client = TestClient(api_module.app)
            resp = client.get("/image_mission/EXPORT_OK/mosaic/export?scale=0.05")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "attachment;" in resp.headers["content-disposition"]

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = set(zf.namelist())
        assert {"mosaic.png", "mosaic.json", "mosaic.pgw"} <= names

        metadata = json.loads(zf.read("mosaic.json").decode("utf-8"))
        assert metadata["mission_id"] == "EXPORT_OK"
        assert metadata["format"] == "png+worldfile+json"
        assert metadata["image_width_px"] == 8
        assert metadata["image_height_px"] == 5
        assert metadata["gsd_m"] == 0.579
        assert metadata["bounding_box"]["TL"]["lat"] == 36.354
        assert metadata["bounding_box"]["BR"]["lon"] == -118.754

        worldfile = zf.read("mosaic.pgw").decode("utf-8").strip().splitlines()
        assert len(worldfile) == 6
        assert float(worldfile[0]) > 0.0
        assert float(worldfile[3]) < 0.0

