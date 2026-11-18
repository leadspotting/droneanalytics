import io
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

import api


def geotagged_jpeg(longitude_seconds):
    image = Image.new('RGB', (160, 120), (100, 160, 90))
    exif = Image.Exif()
    rational = IFDRational
    exif[34853] = {
        1: 'N', 2: (rational(32), rational(4), rational(48)),
        3: 'E', 4: (rational(34), rational(46), rational(longitude_seconds)),
        5: 0, 6: rational(110),
    }
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', exif=exif)
    return buffer.getvalue()


def test_real_exif_upload_stitch_preview_export_and_projection(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.routers.image_missions.TASK3_UPLOADS_DIR", tmp_path)
    saved_missions = api._image_missions.copy()
    saved_cache = api._image_preview_cache.copy()
    try:
        with TestClient(api.app) as client:
            upload = client.post('/image_missions/upload', data={'mission_id': 'SMOKE', 'agl_m': '110'}, files=[
                ('files', ('one.jpg', geotagged_jpeg(48), 'image/jpeg')),
                ('files', ('two.jpg', geotagged_jpeg(49), 'image/jpeg')),
            ])
            assert upload.status_code == 200, upload.text
            assert upload.json()['image_count'] == 2
            metadata = client.get('/image_mission/SMOKE/mosaic')
            assert metadata.status_code == 200
            preview = client.get('/image_mission/SMOKE/mosaic/preview?scale=0.2')
            assert preview.status_code == 200, preview.text[:200] if not preview.is_success else ''
            with Image.open(io.BytesIO(preview.content)) as image:
                assert image.width > 1 and image.height > 1
                assert image.getbbox() is not None
            export = client.get('/image_mission/SMOKE/mosaic/export?scale=0.2')
            assert export.status_code == 200
            with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
                assert {'mosaic.png', 'mosaic.json', 'mosaic.pgw'} <= set(archive.namelist())
            projection = client.get('/image_mission/SMOKE/project?x=0&y=0')
            assert projection.status_code == 200
            assert 31 < projection.json()['lat'] < 33
            assert 33 < projection.json()['lon'] < 35
    finally:
        api._image_missions.clear()
        api._image_missions.update(saved_missions)
        api._image_preview_cache.clear()
        api._image_preview_cache.update(saved_cache)
