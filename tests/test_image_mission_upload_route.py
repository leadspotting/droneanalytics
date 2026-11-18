from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import api as api_module


def test_upload_image_mission_route_registers_uploaded_mission(tmp_path, monkeypatch):
    upload_dir = tmp_path / "task3_uploads"
    monkeypatch.setattr("backend.routers.image_missions.TASK3_UPLOADS_DIR", upload_dir)

    fake_rows = [
        {
            "path": str(upload_dir / "FIELD_UPLOAD" / "DJI_0001.JPG"),
            "lat": 36.1,
            "lon": -118.2,
            "agl_m": 110.0,
            "alt_ellipsoid_m": 1400.0,
            "timestamp": "2024:01:01 10:00:00Z",
            "heading": 0.0,
            "width_px": 4000,
            "height_px": 3000,
            "fov_h": 65.8,
            "fov_v": 46.4,
        }
    ]
    monkeypatch.setattr("backend.routers.image_missions.load_image_dir", lambda directory, agl_m: fake_rows)

    client = TestClient(api_module.app)
    response = client.post(
        "/image_missions/upload",
        data={"mission_id": "field upload", "agl_m": "110"},
        files=[("files", ("DJI_0001.JPG", b"fake-jpg-content", "image/jpeg"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mission_id"] == "FIELD_UPLOAD"
    assert payload["image_count"] == 1
    assert payload["uploaded_files"] == 1
    assert "FIELD_UPLOAD" in api_module._image_missions
    assert Path(payload["path"]).exists()

