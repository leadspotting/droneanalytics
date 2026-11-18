"""Tiny progress/cancellation hook shared by long-running pipeline code.

Pipeline code calls ``report(fraction, stage)`` freely; outside a background
job it does nothing. A job runner installs a reporter with ``install()`` in its
worker thread. ``window(lo, hi)`` maps the 0..1 range of nested code into a
slice of the overall range, so stages can be composed without knowing about
each other.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Optional

Reporter = Callable[[float, Optional[str]], None]


class JobCancelled(Exception):
    """Raised from ``report`` when the running job was cancelled."""


_reporter: contextvars.ContextVar[Optional[Reporter]] = contextvars.ContextVar("progress_reporter", default=None)
_window: contextvars.ContextVar[tuple[float, float]] = contextvars.ContextVar("progress_window", default=(0.0, 1.0))


@contextlib.contextmanager
def install(reporter: Reporter):
    token = _reporter.set(reporter)
    wtoken = _window.set((0.0, 1.0))
    try:
        yield
    finally:
        _window.reset(wtoken)
        _reporter.reset(token)


@contextlib.contextmanager
def window(lo: float, hi: float):
    outer_lo, outer_hi = _window.get()
    span = outer_hi - outer_lo
    token = _window.set((outer_lo + lo * span, outer_lo + hi * span))
    try:
        yield
    finally:
        _window.reset(token)


def report(fraction: float, stage: Optional[str] = None) -> None:
    reporter = _reporter.get()
    if reporter is None:
        return
    lo, hi = _window.get()
    reporter(lo + max(0.0, min(1.0, fraction)) * (hi - lo), stage)
