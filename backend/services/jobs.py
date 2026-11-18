"""Background render jobs with progress and cancellation.

Heavy renders (mosaic preview / export) run one at a time in a single worker
thread, so memory use stays bounded and the API stays responsive. Progress is
reported by the pipeline through ``progress.report``.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import progress
from fastapi import HTTPException
from fastapi.responses import Response

MAX_JOBS_KEPT = 20
MAX_RESULT_BYTES_KEPT = 256 * 1024 * 1024
FINISHED_TTL_S = 30 * 60


@dataclass
class Job:
    id: str
    kind: str
    key: str
    handler: Callable[..., Any]
    kwargs: dict[str, Any]
    status: str = "queued"            # queued | running | done | error | cancelled
    progress: float = 0.0
    stage: str = "Queued"
    error: Optional[str] = None
    status_code: Optional[int] = None
    result: Optional[bytes] = None
    media_type: str = "application/octet-stream"
    headers: dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    cancel_requested: bool = False

    def public(self, queue_position: int = 0) -> dict:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "stage": self.stage,
            "error": self.error,
            "status_code": self.status_code,
            "queue_position": queue_position,
            "result_url": f"/jobs/{self.id}/result" if self.status == "done" else None,
        }


_jobs: "OrderedDict[str, Job]" = OrderedDict()
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render-job")


def _coerce(handler: Callable, params: dict[str, Any]) -> dict[str, Any]:
    """Keep only parameters the handler declares, converted to their types."""
    sig = inspect.signature(handler)
    out: dict[str, Any] = {}
    for name, value in params.items():
        param = sig.parameters.get(name)
        if param is None:
            continue
        ann = param.annotation
        ann_name = ann if isinstance(ann, str) else getattr(ann, "__name__", "")
        try:
            if ann_name == "bool":
                value = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"}
            elif ann_name == "int":
                value = int(value)
            elif ann_name == "float":
                value = float(value)
            elif ann_name == "str":
                value = str(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid value for '{name}'.")
        out[name] = value
    return out


def _run(job: Job) -> None:
    def reporter(fraction: float, stage: Optional[str]) -> None:
        if job.cancel_requested:
            raise progress.JobCancelled()
        job.progress = max(job.progress, fraction)   # monotonic
        if stage:
            job.stage = stage

    if job.cancel_requested:
        job.status, job.finished = "cancelled", time.time()
        return
    job.status, job.stage = "running", "Starting"
    try:
        with progress.install(reporter):
            response = asyncio.run(job.handler(**job.kwargs))
        job.result = bytes(response.body)
        job.media_type = response.media_type or "application/octet-stream"
        job.headers = {k: v for k, v in response.headers.items()
                       if k.lower() in {"content-disposition"}}
        job.progress, job.stage, job.status = 1.0, "Done", "done"
    except progress.JobCancelled:
        job.status, job.stage = "cancelled", "Cancelled"
    except HTTPException as exc:
        job.status, job.error, job.status_code = "error", str(exc.detail), exc.status_code
    except Exception as exc:  # pragma: no cover - defensive
        job.status, job.error, job.status_code = "error", f"{type(exc).__name__}: {exc}", 500
    finally:
        job.finished = time.time()
        _prune()


def _prune() -> None:
    now = time.time()
    with _lock:
        for jid in [j for j, job in _jobs.items()
                    if job.finished and now - job.finished > FINISHED_TTL_S]:
            _jobs.pop(jid, None)
        finished = [j for j, job in _jobs.items() if job.finished]
        total = sum(len(_jobs[j].result or b"") for j in finished)
        while finished and (len(finished) > MAX_JOBS_KEPT or total > MAX_RESULT_BYTES_KEPT):
            oldest = finished.pop(0)
            total -= len(_jobs[oldest].result or b"")
            _jobs.pop(oldest, None)


def submit(kind: str, handler: Callable, params: dict[str, Any], dedupe_key: str) -> Job:
    kwargs = _coerce(handler, params)
    with _lock:
        for job in _jobs.values():
            if job.key == dedupe_key and job.status in {"queued", "running"}:
                return job
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, key=dedupe_key, handler=handler, kwargs=kwargs)
        _jobs[job.id] = job
    _executor.submit(_run, job)
    return job


def get(job_id: str) -> Job:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def queue_position(job: Job) -> int:
    if job.status != "queued":
        return 0
    with _lock:
        ahead = [j for j in _jobs.values() if j.status in {"queued", "running"} and j.created < job.created]
    return len(ahead)


def cancel(job: Job) -> None:
    job.cancel_requested = True
    if job.status == "queued":
        job.status, job.stage, job.finished = "cancelled", "Cancelled", time.time()


def as_response(job: Job) -> Response:
    if job.status != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Job is {job.status}.")
    return Response(content=job.result, media_type=job.media_type, headers=job.headers)
