"""Background render jobs: start, poll progress, fetch result, cancel."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.routers import image_missions
from backend.services import jobs
from backend.state import _image_missions

router = APIRouter(prefix="/jobs")

# kind -> existing route handler (called with the job's parameters)
_HANDLERS = {
    "preview": image_missions.get_image_mosaic_preview,
    "preview_grouped": image_missions.get_image_mosaic_preview_grouped,
    "export": image_missions.export_image_mosaic,
}


class JobRequest(BaseModel):
    kind: str
    mission_id: str
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("")
async def start_job(req: JobRequest):
    handler = _HANDLERS.get(req.kind)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"Unknown job kind '{req.kind}'. Use one of {sorted(_HANDLERS)}.")
    if req.mission_id not in _image_missions:
        raise HTTPException(status_code=404, detail=f"Image mission '{req.mission_id}' not found.")
    params = {**req.params, "mission_id": req.mission_id}
    # `force` only bypasses the cache; it is not part of the identity of the work.
    key = req.kind + json.dumps({k: v for k, v in params.items() if k != "force"}, sort_keys=True, default=str)
    job = jobs.submit(req.kind, handler, params, key)
    return job.public(jobs.queue_position(job))


@router.get("/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    return job.public(jobs.queue_position(job))


@router.get("/{job_id}/result")
async def job_result(job_id: str):
    return jobs.as_response(jobs.get(job_id))


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    job = jobs.get(job_id)
    jobs.cancel(job)
    return job.public()
