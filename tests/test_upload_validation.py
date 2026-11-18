import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    saved = api._missions.copy()
    with TestClient(api.app) as client:
        yield client
    api._missions.clear()
    api._missions.update(saved)


@pytest.mark.parametrize('body', [
    b'',
    b'foo,bar\n1,2\n',
    b'Mission_ID,Latitude,Longitude,Altitude_m\nM,32,34\n',
    b'Mission_ID,Latitude,Longitude,Altitude_m\nM,nan,34,50\n',
    b'Mission_ID,Latitude,Longitude,Altitude_m\nM,91,34,50\n',
    b'Mission_ID,Latitude,Longitude,Altitude_m\n,32,34,50\n',
])
def test_bad_csv_preserves_loaded_missions(client, body):
    saved = api._missions.copy()
    response = client.post('/upload', files={'file': ('bad.csv', body)})
    assert response.status_code == 400
    assert api._missions == saved


def test_csv_with_bom_is_accepted(client):
    body = '\ufeffMission_ID,Latitude,Longitude,Altitude_m\nBOM,32,34,50\n'.encode()
    response = client.post('/upload', files={'file': ('valid.csv', body)})
    assert response.status_code == 200
    assert response.json()['missions'] == ['BOM']
    assert client.get('/mission/BOM/footprints').json()['count'] == 1


@pytest.mark.parametrize('resolution', ['0', '-10', 'nan', 'inf'])
def test_invalid_heatmap_resolution_returns_client_error(client, resolution):
    mission = next(iter(api._missions))
    assert client.get(f'/mission/{mission}/heatmap?resolution_m={resolution}').status_code == 400


def test_empty_srt_preserves_loaded_missions(client):
    saved = api._missions.copy()
    response = client.post('/upload', files={'file': ('bad.srt', b'not telemetry')})
    assert response.status_code == 400
    assert api._missions == saved


@pytest.mark.parametrize('altitude', [0, -10])
def test_invalid_planner_altitude_returns_client_error(client, altitude):
    response = client.post('/plan_mission', json={
        'area': [{'lat': 32, 'lon': 34}, {'lat': 32.001, 'lon': 34}, {'lat': 32, 'lon': 34.001}],
        'altitude_m': altitude,
    })
    assert response.status_code == 400


def test_collinear_polygon_returns_client_error(client):
    response = client.post('/plan_mission', json={
        'area': [{'lat': 32, 'lon': 34}, {'lat': 32.001, 'lon': 34}, {'lat': 32.002, 'lon': 34}],
    })
    assert response.status_code == 400
