from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    missions = api._image_missions.copy()
    with TestClient(api.app) as client:
        yield client
    api._image_missions.clear()
    api._image_missions.update(missions)


def test_invalid_image_replacement_preserves_previous_files_and_registration(tmp_path, monkeypatch, client):
    monkeypatch.setattr("backend.routers.image_missions.TASK3_UPLOADS_DIR", tmp_path)
    old_dir = tmp_path / 'EXISTING'
    old_dir.mkdir()
    old_image = old_dir / 'old.jpg'
    old_image.write_bytes(b'previous good image')
    previous = [{'path': str(old_image), 'lat': 32, 'lon': 34}]
    api._image_missions['EXISTING'] = previous
    monkeypatch.setattr("backend.routers.image_missions.load_image_dir", lambda *args, **kwargs: [])
    response = client.post('/image_missions/upload', data={'mission_id': 'EXISTING'},
                           files=[('files', ('bad.jpg', b'invalid', 'image/jpeg'))])
    assert response.status_code == 400
    assert old_image.read_bytes() == b'previous good image'
    assert api._image_missions['EXISTING'] == previous
    assert list(tmp_path.iterdir()) == [old_dir]


def test_valid_image_replacement_updates_paths_and_removes_old_files(tmp_path, monkeypatch, client):
    monkeypatch.setattr("backend.routers.image_missions.TASK3_UPLOADS_DIR", tmp_path)
    old_dir = tmp_path / 'EXISTING'
    old_dir.mkdir()
    (old_dir / 'old.jpg').write_bytes(b'old')
    monkeypatch.setattr("backend.routers.image_missions.load_image_dir", lambda directory, **kwargs: [{'path': str(directory / 'new.jpg')}])
    response = client.post('/image_missions/upload', data={'mission_id': 'EXISTING'},
                           files=[('files', ('new.jpg', b'new', 'image/jpeg'))])
    assert response.status_code == 200
    assert not (old_dir / 'old.jpg').exists()
    assert Path(api._image_missions['EXISTING'][0]['path']).read_bytes() == b'new'
    assert list(tmp_path.iterdir()) == [old_dir]


def test_failed_image_install_rolls_back_previous_directory(tmp_path, monkeypatch, client):
    monkeypatch.setattr("backend.routers.image_missions.TASK3_UPLOADS_DIR", tmp_path)
    old_dir = tmp_path / 'EXISTING'
    old_dir.mkdir()
    (old_dir / 'old.jpg').write_bytes(b'old')
    monkeypatch.setattr("backend.routers.image_missions.load_image_dir", lambda directory, **kwargs: [{'path': str(directory / 'new.jpg')}])
    rename = Path.rename
    def fail_new_install(path, target):
        if path.name.startswith('.incoming_'):
            raise OSError('simulated rename failure')
        return rename(path, target)
    monkeypatch.setattr(Path, 'rename', fail_new_install)
    with pytest.raises(OSError, match='simulated rename'):
        client.post('/image_missions/upload', data={'mission_id': 'EXISTING'},
                    files=[('files', ('new.jpg', b'new', 'image/jpeg'))])
    assert (old_dir / 'old.jpg').read_bytes() == b'old'


@pytest.mark.parametrize('extension', ['jpg', 'tif'])
def test_bad_reference_replacement_preserves_old_reference(tmp_path, monkeypatch, client, extension):
    monkeypatch.setattr("backend.routers.task4.TASK4_UPLOAD_DIR", tmp_path)
    old_tif = tmp_path / 'reference.tif'
    old_jpg = tmp_path / 'reference.jpg'
    monkeypatch.setattr("backend.routers.task4.TASK4_UPLOAD_TIF", old_tif)
    monkeypatch.setattr("backend.routers.task4.TASK4_UPLOAD_JPG", old_jpg)
    old_tif.write_bytes(b'previous reference')
    response = client.post('/task4/reference/upload', files={'file': (f'bad.{extension}', b'not an image')})
    assert response.status_code == 400
    assert old_tif.read_bytes() == b'previous reference'
    assert list(tmp_path.iterdir()) == [old_tif]
