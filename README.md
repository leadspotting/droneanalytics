# Drone Oversight Forensics System

Drone Oversight is a full-stack drone telemetry and imagery analysis project built around a FastAPI backend and a React/Vite dashboard. It supports mission footprint analysis, coverage heatmaps, mission planning, pixel-to-GPS projection, and EXIF-driven image mission previews.

The integrated version uses the tested image pipeline from GitHub `main`, with
local map-loading, upload, security and packaging fixes. See [REVIEW.md](REVIEW.md)
for verification results and the remaining limits of real-world validation.

## Quick start

**Prerequisites:** Python 3.11+, Node.js 22.12+ (or 20.19+), and an internet connection
for the map tiles.

**Windows:** double-click **RUN.bat**.

1. First run only (a few minutes): it creates `.venv`, installs the Python and npm
   packages.
2. It starts the API (http://localhost:8000) and the UI, and opens
   **http://localhost:5173** in your browser.
3. To stop, close the console window (or press Ctrl+C in it).

**Any other platform:**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm ci
python run.py --dev                # API + UI, open http://localhost:5173
```

Launcher modes (`run.py`):

| Command | What it does |
|---|---|
| `python run.py --dev` | API with auto-reload + Vite dev server (UI on :5173). What `RUN.bat` uses. |
| `python run.py` | App mode: builds the UI if needed and serves everything from http://localhost:8000. |
| `python run.py --api` | Backend only (used internally by `npm start`). |
| add `--no-browser` | Do not open a browser window. |

## Using the dashboard

1. **Telemetry missions:** the bundled sample (`Drone_Forensics_Dataset.csv`) loads on start.
   Pick a mission in the header to see its flight path, camera footprints and coverage
   heatmap. *Upload CSV* replaces the data with your own CSV (columns `Mission_ID`,
   `Latitude`, `Longitude`, `Altitude_m`, optional `Heading`) or a DJI `.srt` file.
2. **Mission planning:** draw a polygon on the map; the planner returns a lawnmower flight
   path. Choose a drone to get a feasibility check (flight time vs. battery). Areas that would
   need more than 50,000 waypoints are rejected with a hint to shrink the area or raise the altitude.
3. **Image missions (mosaic):** in the sidebar, upload geotagged JPGs (GPS EXIF required).
   The mosaic renders in the background; a progress bar with a Cancel button shows the stage.
   *Re-render* recomputes it, *Download Export* gives a ZIP (PNG + georeference files).
4. **Projection:** click the preview (or enter X,Y) to convert pixels to GPS and back.

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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python not found` / `Node.js not found` in RUN.bat | Install them and make sure they are on PATH (new console window). |
| Port 8000 or 5173 already in use | Close the previous run (its console window) or the other program using the port. |
| UI loads but shows no data / 502 | The API is still starting (first start imports heavy libraries); wait ~10 s and reload. |
| Map is grey | No internet access: basemap tiles are loaded online. |
| First run seems stuck | `pip install` / `npm ci` are downloading; watch the console. |
| Upload rejected | CSV/SRT max 50 MB; image uploads max 2000 files / 2 GB. |

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

### Background render jobs

Heavy mosaic renders run as jobs (one at a time) so the API stays responsive.

- `POST /jobs` with `{"kind": "preview" | "preview_grouped" | "export", "mission_id": "...", "params": {"scale": 0.12}}`
  returns `{job_id, status, progress, stage}`
- `GET /jobs/{job_id}` — status (`queued|running|done|error|cancelled`), `progress` 0..1, `stage`
- `GET /jobs/{job_id}/result` — the PNG (preview) or ZIP (export) once `done`
- `DELETE /jobs/{job_id}` — cancel

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
- Map basemaps need an internet connection. Leaflet styles are bundled locally; dark maps use Stadia Maps, light maps OpenStreetMap, satellite maps Esri and topographic maps OpenTopoMap. Changing styles keeps mission overlays and the current view. Run this application on loopback as configured; it has no user authentication.
- Image missions and Task 4 references are validated before replacing existing files. An invalid replacement preserves the previous upload.
- `build/`, `dist/`, `release/`, `node_modules/`, credentials and local planning documents are excluded by `.gitignore`. Keep the `.spec`, lockfile, source, tests, and sample CSVs in Git. Distribute executables separately as GitHub Release assets.
- The SEKI image dataset is spatially split into separate capture groups. A grouped preview is the correct expected visualization.
- `refine=true` on Task 3 preview routes is experimental and is not the supported delivery path.
