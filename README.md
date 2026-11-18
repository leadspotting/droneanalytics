# Drone Oversight Forensics System

Drone Oversight is a full-stack drone telemetry and imagery analysis project built around a FastAPI backend and a React/Vite dashboard. It supports mission footprint analysis, coverage heatmaps, mission planning, pixel-to-GPS projection, and EXIF-driven image mission previews.

The integrated version uses the tested image pipeline from GitHub `main`, with
local map-loading, upload, security and packaging fixes. See [REVIEW.md](REVIEW.md)
for verification results and the remaining limits of real-world validation.

## Run

Windows: double-click **RUN.bat** (developer mode: opens http://localhost:5173). On first run it creates `.venv`, installs the
Python and npm packages and starts the API with auto-reload plus the Vite UI.

Any platform (Python 3.11+, Node.js 22.12+ / 20.19+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py            # app mode: one port (8000), opens the browser
python run.py --dev      # developer mode: API auto-reload + Vite on :5173
```

## Project layout

```
run.py                 single launcher (app / --dev / --api)
api.py                 compatibility shim for `uvicorn api:app`
backend/
  main.py              app assembly
  config.py            paths and resource limits
  state.py             in-memory stores (+ byte-bounded LRU preview cache)
  routers/             missions, planner, drones, image_missions, task4
  services/            telemetry, images, task4, drones helpers
dashboard/             React + Vite UI
*.py (root)            domain/algorithm modules (footprints, heatmap, stitching ...)
```

Resource limits (see `backend/config.py`): CSV/SRT upload 50 MB, 2000 images / 2 GB per
image upload, 256 MB preview cache, 250k heatmap cells, 40 MP mosaic canvas
(larger canvases are coarsened and reported as `canvas_downscaled`).

## Standalone Windows build

`build_exe.bat` builds the UI and packages `run.py` with PyInstaller into
`release/DroneOversight`; double-click `DroneOversight.exe` to run.

## Quick Manual Check

1. Open `http://localhost:5173`.
2. Verify a mission loads and the map renders.
3. Select a mission and confirm footprints and heatmap appear.
4. Draw a polygon and confirm the planner returns an orange flight path.
5. Load an image mission in the sidebar and confirm a mosaic preview appears.
6. Use the projection panel to convert pixel coordinates to GPS and back.

## API Summary

### Missions

- `GET /missions`
- `POST /upload`
- `GET /mission/{mission_id}/footprints`
- `GET /mission/{mission_id}/heatmap?resolution_m=200`
- `GET /mission/{mission_id}/heatmap?resolution_m=200&drone_id=DRN_A`
- `GET /mission/{mission_id}/mosaic`
- `GET /mission/{mission_id}/project?x=0&y=0`
- `GET /mission/{mission_id}/project/reverse?lat=...&lon=...`

Example heatmap response:

```json
{
  "mission_id": "HEATMAP_MULTI",
  "drone_filter_applied": "DRN_A",
  "coverage_pct": 12.5,
  "max_overlap": 1,
  "resolution_m": 150.0,
  "cells": [
    { "lat": 34.0504, "lon": 118.2505, "count": 1 }
  ]
}
```

### Mission Planner

- `POST /plan_mission`
- Optional planner constraint inputs:
  - `drone_id`
  - `cruise_speed_kmh`
  - `safe_flight_time_min`
- When constraint inputs are supplied, the response includes a `feasibility` object.
- When no constraint inputs are supplied, the response stays backward-compatible and does not include `feasibility`.

Example request:

```json
{
  "area": [
    { "lat": 32.0800, "lon": 34.7800 },
    { "lat": 32.0800, "lon": 34.7810 },
    { "lat": 32.0810, "lon": 34.7810 },
    { "lat": 32.0810, "lon": 34.7800 }
  ],
  "altitude_m": 100.0,
  "overlap_h": 0.7,
  "overlap_v": 0.7
}
```

Example constrained request:

```json
{
  "area": [
    { "lat": 32.0800, "lon": 34.7800 },
    { "lat": 32.0800, "lon": 34.7810 },
    { "lat": 32.0810, "lon": 34.7810 },
    { "lat": 32.0810, "lon": 34.7800 }
  ],
  "altitude_m": 100.0,
  "overlap_h": 0.7,
  "overlap_v": 0.7,
  "drone_id": "mavic-4-pro"
}
```

### Image Missions

- `POST /image_missions/load`
- `GET /image_missions`
- `GET /image_mission/{mission_id}/footprints`
- `GET /image_mission/{mission_id}/mosaic`
- `GET /image_mission/{mission_id}/mosaic/preview`
- `GET /image_mission/{mission_id}/mosaic/preview/grouped`
- `GET /image_mission/{mission_id}/project?x=0&y=0`
- `GET /image_mission/{mission_id}/project/reverse?lat=...&lon=...`

## Validation Commands

Run the critical Python route tests:

```powershell
python -m pytest tests/test_heatmap.py tests/test_heatmap_route.py tests/test_mosaic_route.py tests/test_project_route.py -q
```

Run the full Python test suite:

```powershell
python -m pytest -q
```

Build the frontend:

```powershell
npm run build
npm test
```

## Notes

- The dashboard uses an in-memory backend store. A valid telemetry upload replaces the current mission dataset for the session; repeated uploads refresh the map even when the mission ID is unchanged.
- Only a valid CSV/SRT upload replaces telemetry; rejected uploads preserve it. Custom drones and uploaded mission registrations are also reset on restart.
- `task3_data/` and `task4_data/` are optional local datasets, excluded from Git. Real-data tests skip when their fixtures are absent. No imagery is bundled in a source checkout.
- Map basemaps need an internet connection. Leaflet styles are bundled locally; light/dark maps use CARTO and satellite/topographic maps use Esri. Changing styles keeps mission overlays and the current view. Run this application on loopback as configured; it has no user authentication.
- Image missions and Task 4 references are validated before replacing existing files. An invalid replacement preserves the previous upload.
- `build/`, `dist/`, `release/`, `node_modules/`, credentials and local planning documents are excluded by `.gitignore`. Keep the `.spec`, lockfile, source, tests, and sample CSVs in Git. Distribute executables separately as GitHub Release assets.
- The SEKI image dataset is spatially split into separate capture groups. A grouped preview is the correct expected visualization.
- `refine=true` on Task 3 preview routes is experimental and is not the supported delivery path.
