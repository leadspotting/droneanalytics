"""Process-wide in-memory stores.

All stores are mutated **in place** (never re-bound), so every module that
imported them keeps seeing the same object.
"""
from __future__ import annotations

from collections import OrderedDict

from backend.config import PREVIEW_CACHE_MAX_BYTES, PREVIEW_CACHE_MAX_ENTRIES

# mission_id -> list of telemetry rows (CSV / SRT missions)
_missions: dict[str, list[dict]] = {}

# mission_id -> list of image rows (path, lat, lon, agl_m, ...)
_image_missions: dict[str, list[dict]] = {}

# user-defined drones added through the UI
_custom_drones: list[dict] = []


class ByteLRUCache(OrderedDict):
    """LRU cache bounded by total payload bytes and entry count.

    Values are ``bytes``. The least recently used entries are evicted when
    either limit is exceeded; a single value larger than the byte limit is
    simply not cached.
    """

    max_bytes = PREVIEW_CACHE_MAX_BYTES
    max_entries = PREVIEW_CACHE_MAX_ENTRIES

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        return default

    def __setitem__(self, key, value):
        if len(value) > self.max_bytes:
            self.pop(key, None)
            return
        super().__setitem__(key, value)
        self.move_to_end(key)
        total = sum(len(v) for v in self.values())
        while self and (total > self.max_bytes or len(self) > self.max_entries):
            _, evicted = self.popitem(last=False)
            total -= len(evicted)


# (kind, mission_id, ...) -> PNG bytes
_image_preview_cache: ByteLRUCache = ByteLRUCache()
