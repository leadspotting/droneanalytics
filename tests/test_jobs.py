import time

import pytest
from fastapi.testclient import TestClient

import api
import progress
from tests.test_image_pipeline_smoke import geotagged_jpeg


def _wait(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f'/jobs/{job_id}').json()
        if status['status'] in {'done', 'error', 'cancelled'}:
            return status
        time.sleep(0.05)
    raise AssertionError('job did not finish')


@pytest.fixture
def mission(tmp_path, monkeypatch):
    monkeypatch.setattr('backend.routers.image_missions.TASK3_UPLOADS_DIR', tmp_path)
    saved = api._image_missions.copy()
    with TestClient(api.app) as client:
        response = client.post('/image_missions/upload', data={'mission_id': 'JOBM', 'agl_m': '110'}, files=[
            ('files', ('one.jpg', geotagged_jpeg(48), 'image/jpeg')),
            ('files', ('two.jpg', geotagged_jpeg(49), 'image/jpeg')),
        ])
        assert response.status_code == 200, response.text
        yield client
    api._image_missions.clear()
    api._image_missions.update(saved)
    api._image_preview_cache.clear()


def test_preview_job_reports_progress_and_returns_png(mission):
    started = mission.post('/jobs', json={'kind': 'preview', 'mission_id': 'JOBM',
                                          'params': {'scale': 0.2, 'force': True}})
    assert started.status_code == 200, started.text
    status = _wait(mission, started.json()['job_id'])
    assert status['status'] == 'done', status
    assert status['progress'] == 1.0
    result = mission.get(status['result_url'])
    assert result.status_code == 200
    assert result.content[:8] == b'\x89PNG\r\n\x1a\n'


def test_export_job_returns_zip(mission):
    job = mission.post('/jobs', json={'kind': 'export', 'mission_id': 'JOBM', 'params': {'scale': 0.2}}).json()
    status = _wait(mission, job['job_id'])
    assert status['status'] == 'done', status
    assert mission.get(status['result_url']).content[:2] == b'PK'


def test_unknown_mission_and_kind_rejected(mission):
    assert mission.post('/jobs', json={'kind': 'preview', 'mission_id': 'NOPE'}).status_code == 404
    assert mission.post('/jobs', json={'kind': 'bogus', 'mission_id': 'JOBM'}).status_code == 400
    assert mission.get('/jobs/doesnotexist').status_code == 404


def test_result_before_done_is_conflict_and_cancel_works(mission):
    job = mission.post('/jobs', json={'kind': 'preview', 'mission_id': 'JOBM',
                                      'params': {'scale': 0.2, 'force': True, 'normalize_altitude': False}}).json()
    mission.delete(f"/jobs/{job['job_id']}")
    status = _wait(mission, job['job_id'])
    assert status['status'] in {'cancelled', 'done'}
    if status['status'] == 'cancelled':
        assert mission.get(f"/jobs/{job['job_id']}/result").status_code == 409


def test_progress_window_maps_nested_ranges():
    seen = []
    with progress.install(lambda f, s: seen.append(round(f, 3))):
        with progress.window(0.4, 0.95):
            progress.report(0.0)
            progress.report(1.0)
    progress.report(0.5)  # outside a job: no-op
    assert seen == [0.4, 0.95]
