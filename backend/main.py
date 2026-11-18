"""FastAPI application assembly."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import DIST_DIR
from backend.routers import drones, image_missions, jobs, missions, planner, task4
from backend.services.images import _autoload_image_missions
from backend.services.telemetry import _load_default

app = FastAPI(title="Drone Oversight Dashboard", version="1.0")

if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

for _module in (missions, planner, drones, image_missions, task4, jobs):
    app.include_router(_module.router)

_load_default()
_autoload_image_missions()


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = DIST_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not built yet. Please run 'npm run build'.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
