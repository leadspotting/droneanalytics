"""
coverage_heatmap.py  —  Task 2
================================
Coverage Heatmap: builds a 2-D grid over an area of interest (AOI) and
tracks how many drone image footprints (Task 1) cover each cell.

Math summary
------------
* **Grid construction** (flat-Earth approximation):
    height_m = Δlat × 111_320
    width_m  = Δlon × 111_320 × cos(mid_lat)
  Grid is rows × cols of equal-area cells, each `resolution_m` on a side.

* **GPS → local metres**: convert every GPS point to a (x_m, y_m) offset
  from the grid origin (SW corner), using the same flat-Earth formulas.

* **Axis-aligned bounding box (AABB) pre-filter**: for each footprint,
  compute its AABB in grid-row/col indices so we only test candidate cells,
  not the entire grid (O(footprint_area / cell_area) instead of O(grid)).

* **Ray-casting point-in-polygon**: for a cell centre (cx, cy) and a
  convex polygon (the footprint corners), cast a ray in the +X direction
  and count edge crossings.  Odd count ⟹ inside.  This works for any
  convex or concave polygon with no self-intersections.

* **Overlap count**: a cell's integer value equals how many footprints cover
  it, enabling easy overlap / redundancy analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np

from models import GPSCoordinate
from footprint_calculator import FootprintResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRES_PER_LAT_DEG: float = 111_320.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gps_to_metres(
    coord: GPSCoordinate,
    origin: GPSCoordinate,
    cos_lat: float,
) -> tuple[float, float]:
    """
    Convert a GPS coordinate to local (x_m, y_m) East-North metres
    relative to *origin*.

    Parameters
    ----------
    coord   : GPS point to convert.
    origin  : Reference GPS point (grid SW corner).
    cos_lat : cos(origin.lat in radians) — pre-computed for speed.

    Returns
    -------
    (x_m, y_m)  — East and North offsets in metres.
    """
    y_m = (coord.lat - origin.lat) * METRES_PER_LAT_DEG
    x_m = (coord.lon - origin.lon) * METRES_PER_LAT_DEG * cos_lat
    return x_m, y_m


def _point_in_polygon(px: float, py: float, polygon: np.ndarray) -> bool:
    """
    Ray-casting point-in-polygon test (works for any simple polygon).

    Parameters
    ----------
    px, py  : Point to test (metres).
    polygon : (N, 2) array of (x, y) vertices in order (open or closed).

    Returns
    -------
    bool  — True if (px, py) is strictly inside or on the boundary.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Check if the horizontal ray from (px, py) crosses edge (j→i)
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CoverageHeatmap:
    """
    Task 2: 2-D coverage grid over a rectangular area of interest.

    Each cell counts how many drone image footprints (from Task 1) cover it.

    Parameters
    ----------
    area_corners : list[GPSCoordinate]
        Four GPS corners of the AOI.  The class computes the bounding box
        automatically — corners can be in any order.
    grid_resolution_m : float
        Side length of each square grid cell in metres (default 10 m).

    Example
    -------
    >>> heatmap = CoverageHeatmap(area_corners, grid_resolution_m=5.0)
    >>> heatmap.add_footprint(footprint_result)
    >>> print(heatmap.coverage_percentage())
    """

    def __init__(
        self,
        area_corners: list[GPSCoordinate],
        grid_resolution_m: float = 10.0,
    ) -> None:
        if len(area_corners) < 3:
            raise ValueError("area_corners must have at least 3 GPS points.")
        if grid_resolution_m <= 0:
            raise ValueError("grid_resolution_m must be positive.")

        self.grid_resolution_m = grid_resolution_m

        # ── Bounding box ────────────────────────────────────────────────
        lats = [c.lat for c in area_corners]
        lons = [c.lon for c in area_corners]
        self._min_lat = min(lats)
        self._max_lat = max(lats)
        self._min_lon = min(lons)
        self._max_lon = max(lons)

        # SW corner = grid origin
        self._origin = GPSCoordinate(lat=self._min_lat, lon=self._min_lon)
        self._mid_lat_rad = math.radians((self._min_lat + self._max_lat) / 2.0)
        self._cos_lat = math.cos(self._mid_lat_rad)

        # ── Grid dimensions ─────────────────────────────────────────────
        height_m = (self._max_lat - self._min_lat) * METRES_PER_LAT_DEG
        width_m  = (self._max_lon - self._min_lon) * METRES_PER_LAT_DEG * self._cos_lat

        self._rows = max(1, math.ceil(height_m / grid_resolution_m))
        self._cols = max(1, math.ceil(width_m  / grid_resolution_m))

        # ── Hit-count array ─────────────────────────────────────────────
        self._grid: np.ndarray = np.zeros((self._rows, self._cols), dtype=np.int32)

        # Store footprints for inspection
        self._footprints: list[FootprintResult] = []

    # ------------------------------------------------------------------
    # Core method: add a footprint
    # ------------------------------------------------------------------

    def add_footprint(self, footprint: FootprintResult) -> None:
        """
        Register a new image footprint (from Task 1) into the grid.

        Algorithm
        ---------
        1. Convert the 4 footprint GPS corners to local metres.
        2. Build the AABB in grid indices for quick pre-filtering.
        3. For each candidate cell centre, run the ray-casting test.
        4. Increment the hit counter for cells inside the footprint.

        Parameters
        ----------
        footprint : FootprintResult
            Output of ``FootprintCalculator.compute()``.
        """
        self._footprints.append(footprint)

        # Step 1 — convert footprint corners to (x_m, y_m)
        corner_metres = np.array([
            _gps_to_metres(c, self._origin, self._cos_lat)
            for c in footprint.corners
        ])  # shape (4, 2)

        # Step 2 — AABB in grid-cell indices (clipped to grid bounds)
        xs, ys = corner_metres[:, 0], corner_metres[:, 1]
        col_min = max(0, int(math.floor(xs.min() / self.grid_resolution_m)))
        col_max = min(self._cols - 1, int(math.ceil(xs.max()  / self.grid_resolution_m)))
        row_min = max(0, int(math.floor(ys.min() / self.grid_resolution_m)))
        row_max = min(self._rows - 1, int(math.ceil(ys.max()  / self.grid_resolution_m)))

        # Step 3 & 4 — ray-casting for each candidate cell
        half = self.grid_resolution_m / 2.0
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                # Centre of this cell in metres
                cx = (col + 0.5) * self.grid_resolution_m
                cy = (row + 0.5) * self.grid_resolution_m
                if _point_in_polygon(cx, cy, corner_metres):
                    self._grid[row, col] += 1

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_grid(self) -> np.ndarray:
        """
        Return the raw hit-count grid (read-only copy).

        Returns
        -------
        np.ndarray  shape (rows, cols), dtype int32
            Each entry = number of footprints covering that cell.
        """
        return self._grid.copy()

    def coverage_percentage(self) -> float:
        """
        Fraction of the AOI covered by at least one footprint (0.0–100.0 %).

        Returns
        -------
        float  — percentage in [0.0, 100.0].
        """
        total = self._grid.size
        if total == 0:
            return 0.0
        covered = int(np.count_nonzero(self._grid))
        return 100.0 * covered / total

    def max_overlap(self) -> int:
        """Return the highest hit count across all cells (maximum redundancy)."""
        return int(self._grid.max())

    def get_cell_center(self, row: int, col: int) -> GPSCoordinate:
        """
        Return the GPS coordinate of the centre of grid cell (row, col).

        Used by Task 6 (Mission Planner) to get ground-truth positions.

        Parameters
        ----------
        row : int   0 = southernmost row.
        col : int   0 = westernmost column.

        Returns
        -------
        GPSCoordinate  with alt=0.0.

        Raises
        ------
        IndexError if row/col are out of bounds.
        """
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise IndexError(
                f"Cell ({row}, {col}) is out of bounds "
                f"(grid is {self._rows}×{self._cols})."
            )
        x_m = (col + 0.5) * self.grid_resolution_m
        y_m = (row + 0.5) * self.grid_resolution_m
        lat = self._origin.lat + y_m / METRES_PER_LAT_DEG
        lon = self._origin.lon + x_m / (METRES_PER_LAT_DEG * self._cos_lat)
        return GPSCoordinate(lat=lat, lon=lon, alt=0.0)

    def covered_gps_points(self) -> Generator[GPSCoordinate, None, None]:
        """
        Generator yielding the GPS centre of every covered cell.

        Useful for visualising which spots on the ground were photographed.
        """
        rows, cols = np.nonzero(self._grid)
        for r, c in zip(rows, cols):
            yield self.get_cell_center(int(r), int(c))

    def uncovered_gps_points(self) -> Generator[GPSCoordinate, None, None]:
        """
        Generator yielding the GPS centre of every uncovered cell.

        Useful for Task 6 (Mission Planner) to identify gaps.
        """
        rows, cols = np.where(self._grid == 0)
        for r, c in zip(rows, cols):
            yield self.get_cell_center(int(r), int(c))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols) of the grid."""
        return (self._rows, self._cols)

    @property
    def num_footprints(self) -> int:
        """Number of footprints registered so far."""
        return len(self._footprints)

    def __repr__(self) -> str:
        return (
            f"CoverageHeatmap("
            f"grid={self._rows}×{self._cols}, "
            f"res={self.grid_resolution_m}m, "
            f"footprints={self.num_footprints}, "
            f"coverage={self.coverage_percentage():.1f}%)"
        )


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models import CameraConfig, DroneTelemetry, GimbalMode
    from footprint_calculator import FootprintCalculator

    print("=== Coverage Heatmap - Self-Test ===\n")

    # Define a small 500 m × 500 m AOI over Tel Aviv
    SW = GPSCoordinate(lat=32.0800, lon=34.7780)
    NE = GPSCoordinate(lat=32.0845, lon=34.7836)
    area = [
        SW,
        GPSCoordinate(lat=SW.lat, lon=NE.lon),
        NE,
        GPSCoordinate(lat=NE.lat, lon=SW.lon),
    ]

    heatmap = CoverageHeatmap(area, grid_resolution_m=10.0)
    print(f"Grid created: {heatmap.shape[0]} rows × {heatmap.shape[1]} cols")
    print(f"Initial coverage: {heatmap.coverage_percentage():.1f}%\n")

    # Drone 1 — centre of AOI, 80 m altitude, 90°×60° FOV, heading North
    tel1 = DroneTelemetry(
        position=GPSCoordinate(lat=32.0822, lon=34.7808, alt=80.0),
        heading=0.0,
        camera=CameraConfig(fov_h=90.0, fov_v=60.0, gimbal_mode=GimbalMode.NADIR),
    )
    fp1 = FootprintCalculator(tel1).compute()
    heatmap.add_footprint(fp1)
    print(f"After footprint 1:  coverage = {heatmap.coverage_percentage():.1f}%  "
          f"(footprint {fp1.ground_width_m:.0f}m × {fp1.ground_height_m:.0f}m)")

    # Drone 2 — offset East, same altitude, creates overlap with fp1
    tel2 = DroneTelemetry(
        position=GPSCoordinate(lat=32.0822, lon=34.7825, alt=80.0),
        heading=0.0,
        camera=CameraConfig(fov_h=90.0, fov_v=60.0, gimbal_mode=GimbalMode.NADIR),
    )
    fp2 = FootprintCalculator(tel2).compute()
    heatmap.add_footprint(fp2)
    print(f"After footprint 2:  coverage = {heatmap.coverage_percentage():.1f}%  "
          f"max overlap = {heatmap.max_overlap()}x")

    print(f"\nFinal heatmap: {heatmap}")
    print(f"\nGrid (rows=S→N, cols=W→E):\n{heatmap.get_grid()}")
