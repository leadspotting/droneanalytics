"""
mission_planner.py  —  Task 6
===============================
Mission Planner: Generates an optimal waypoint flight path to cover a given area.

Given a polygon area of interest and camera/flight parameters, this
produces an ordered list of GPS waypoints for the drone to follow so
that the entire area is photographed with the specified overlap.
"""

from __future__ import annotations

import math

from models import GPSCoordinate, CameraConfig

METRES_PER_LAT_DEG = 111320.0


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _polygon_signed_area(polygon: list[GPSCoordinate]) -> float:
    """
    Shoelace formula for the signed area of a polygon in (lon, lat) space.

    Returns a non-zero value for a valid polygon. A result near zero indicates
    degenerate (collinear) vertices.
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i].lon * polygon[j].lat
        area -= polygon[j].lon * polygon[i].lat
    return area / 2.0


def _scanline_intersections(lat: float, polygon: list[GPSCoordinate]) -> list[float]:
    """
    Return a sorted list of intersection longitudes for a horizontal scan line
    at the given latitude.

    Uses the top-left rule: an edge (P_i → P_{i+1}) is counted when
        (lat_i ≤ lat < lat_{i+1})  OR  (lat_{i+1} ≤ lat < lat_i)
    This avoids double-counting shared vertices and skips horizontal edges,
    ensuring the output list always has an even length for a simple polygon.
    """
    n = len(polygon)
    intersections: list[float] = []
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        lat1, lon1 = p1.lat, p1.lon
        lat2, lon2 = p2.lat, p2.lon
        # Top-left rule: edge counted iff it strictly brackets lat from below
        if (lat1 <= lat < lat2) or (lat2 <= lat < lat1):
            t = (lat - lat1) / (lat2 - lat1)
            lon_x = lon1 + t * (lon2 - lon1)
            intersections.append(lon_x)
    return sorted(intersections)


def _waypoints_along_segment(
    lat: float,
    lon_start: float,
    lon_end: float,
    delta_lon: float,
    alt: float,
) -> list[GPSCoordinate]:
    """
    Place evenly-spaced waypoints within [lon_start, lon_end].

    The first waypoint is offset half a step inside the segment so the drone's
    camera footprint is centred within the strip.

    After the regular grid is placed, a closing waypoint is inserted when the
    last footprint's right edge falls short of lon_end (i.e., when the segment
    width is not an exact multiple of delta_lon). The closing waypoint is
    positioned so its right footprint edge aligns exactly with lon_end.

    If the segment is too narrow for even one half-step waypoint, a single
    waypoint is placed at the segment midpoint so the area is not silently
    skipped.
    """
    wps: list[GPSCoordinate] = []
    lon = lon_start + delta_lon / 2.0
    while lon <= lon_end:
        wps.append(GPSCoordinate(lat=lat, lon=lon, alt=alt))
        lon += delta_lon

    if wps:
        # If the last footprint's right edge doesn't reach lon_end, add a
        # closing waypoint whose right edge aligns with lon_end.
        right_edge = wps[-1].lon + delta_lon / 2.0
        if right_edge < lon_end - 1e-10:
            closing_lon = lon_end - delta_lon / 2.0
            if closing_lon > wps[-1].lon + 1e-10:  # guard against float duplicates
                wps.append(GPSCoordinate(lat=lat, lon=closing_lon, alt=alt))
    else:
        # Segment narrower than half a footprint width — cover with one centre point
        wps.append(GPSCoordinate(lat=lat, lon=(lon_start + lon_end) / 2.0, alt=alt))
    return wps


# ---------------------------------------------------------------------------
# MissionPlanner
# ---------------------------------------------------------------------------

class MissionPlanner:
    """
    Task 6: Generates an optimal waypoint flight path to cover a given area.

    Inputs
    ------
    area_polygon  : list[GPSCoordinate]   — ordered GPS vertices of the area
                    (simple, non-self-intersecting polygon; ≥ 3 points)
    altitude_m    : float                 — planned flight altitude (AGL, metres)
    camera_config : CameraConfig
    overlap_h     : float                 — horizontal overlap fraction (0.0–1.0)
    overlap_v     : float                 — vertical overlap fraction (0.0–1.0)
    """

    def __init__(
        self,
        area_polygon: list[GPSCoordinate],
        altitude_m: float,
        camera_config: CameraConfig,
        overlap_h: float = 0.7,
        overlap_v: float = 0.7,
    ) -> None:
        if len(area_polygon) < 3:
            raise ValueError("area_polygon must contain at least 3 GPS points.")
        if altitude_m <= 0:
            raise ValueError("altitude_m must be positive.")
        if abs(_polygon_signed_area(area_polygon)) < 1e-12:
            raise ValueError(
                "area_polygon has effectively zero area (degenerate or collinear vertices)."
            )

        self.area_polygon = area_polygon
        self.altitude_m = altitude_m
        self.camera_config = camera_config
        self.overlap_h = max(0.0, min(overlap_h, 0.99))
        self.overlap_v = max(0.0, min(overlap_v, 0.99))

    def _compute_strip_spacing_m(self) -> tuple[float, float]:
        """
        Compute the along-track and cross-track spacing between image centres,
        accounting for the desired overlap percentages.

        Returns
        -------
        (along_track_m, cross_track_m)
        """
        fov_h_rad = math.radians(self.camera_config.fov_h)
        fov_v_rad = math.radians(self.camera_config.fov_v or self.camera_config.fov_h)

        footprint_width_m = 2.0 * self.altitude_m * math.tan(fov_h_rad / 2.0)
        footprint_height_m = 2.0 * self.altitude_m * math.tan(fov_v_rad / 2.0)

        along_track_m = footprint_height_m * (1.0 - self.overlap_v)
        cross_track_m = footprint_width_m * (1.0 - self.overlap_h)

        return along_track_m, cross_track_m

    def estimate_waypoint_count(self) -> int:
        """Cheap upper-ish estimate of how many waypoints plan() would produce."""
        along_track_m, cross_track_m = self._compute_strip_spacing_m()
        if along_track_m <= 0 or cross_track_m <= 0:
            return 0
        lats = [pt.lat for pt in self.area_polygon]
        cos_lat = math.cos(math.radians((min(lats) + max(lats)) / 2.0))
        area_m2 = (
            abs(_polygon_signed_area(self.area_polygon))
            * METRES_PER_LAT_DEG ** 2 * max(cos_lat, 1e-6)
        )
        return int(area_m2 / (along_track_m * cross_track_m)) + 1

    def plan(self) -> list[GPSCoordinate]:
        """
        Generate a lawnmower (boustrophedon) waypoint pattern that covers the
        polygon area.

        The algorithm sweeps horizontal scan lines from south to north in steps
        of `delta_lat`. For each scan line, polygon edge intersections are
        computed to find the exact longitude ranges that lie inside the polygon.
        Waypoints are placed only within those ranges, so no waypoint is emitted
        outside the area of interest — including for non-rectangular polygons.

        Concave polygons with multiple disjoint intersection segments per strip
        are handled naturally: each segment is processed in serpentine order.

        Returns
        -------
        list[GPSCoordinate]
            Ordered list of GPS waypoints (drone flies them in sequence).
        """
        along_track_m, cross_track_m = self._compute_strip_spacing_m()

        lats = [pt.lat for pt in self.area_polygon]
        min_lat, max_lat = min(lats), max(lats)

        mid_lat_rad = math.radians((min_lat + max_lat) / 2.0)
        cos_lat = math.cos(mid_lat_rad)

        delta_lat = along_track_m / METRES_PER_LAT_DEG
        delta_lon = cross_track_m / (METRES_PER_LAT_DEG * cos_lat)

        waypoints: list[GPSCoordinate] = []

        def _add_strip(lat: float, idx: int) -> None:
            """Compute intersections at lat and append waypoints to outer list."""
            xs = _scanline_intersections(lat, self.area_polygon)
            # xs must be even-length for a simple polygon; skip malformed strips
            if len(xs) < 2 or len(xs) % 2 != 0:
                return
            segments = [(xs[i], xs[i + 1]) for i in range(0, len(xs), 2)]
            moving_east = (idx % 2 == 0)
            if not moving_east:
                segments = list(reversed(segments))
            for seg_start, seg_end in segments:
                strip_wps = _waypoints_along_segment(
                    lat, seg_start, seg_end, delta_lon, self.altitude_m
                )
                if not moving_east:
                    strip_wps = list(reversed(strip_wps))
                waypoints.extend(strip_wps)

        strip_index = 0
        current_lat = min_lat + delta_lat / 2.0
        last_strip_lat: float | None = None

        while current_lat <= max_lat:
            _add_strip(current_lat, strip_index)
            last_strip_lat = current_lat
            current_lat += delta_lat
            strip_index += 1

        if last_strip_lat is None:
            # Polygon height < delta_lat / 2: the regular grid produced nothing.
            # Place one strip at the vertical midpoint so small-area polygons are covered.
            _add_strip((min_lat + max_lat) / 2.0, strip_index)
        else:
            # Closing northern strip: if the last strip's top footprint edge falls
            # short of max_lat (polygon height is not an exact multiple of delta_lat),
            # add one final strip centred so its top edge aligns with max_lat.
            top_edge = last_strip_lat + delta_lat / 2.0
            if top_edge < max_lat - 1e-10:
                closing_lat = max_lat - delta_lat / 2.0
                if closing_lat > last_strip_lat + 1e-10:  # guard against float duplicates
                    _add_strip(closing_lat, strip_index)

        return waypoints

    def compute_route_distance(self, waypoints: list[GPSCoordinate] | None = None) -> float:
        """
        Compute the total route distance in metres for the planned waypoint path.

        Uses the same flat-earth approximation as the frontend computeRouteDistance
        function so feasibility calculations are consistent across backend and frontend.

        Returns
        -------
        float   Total path length in metres.
        """
        if waypoints is None:
            waypoints = self.plan()
        if len(waypoints) < 2:
            return 0.0

        total_distance_m = 0.0
        for i in range(len(waypoints) - 1):
            pt1 = waypoints[i]
            pt2 = waypoints[i + 1]
            mid_lat_rad = math.radians((pt1.lat + pt2.lat) / 2.0)
            dy_m = (pt2.lat - pt1.lat) * METRES_PER_LAT_DEG
            dx_m = (pt2.lon - pt1.lon) * METRES_PER_LAT_DEG * math.cos(mid_lat_rad)
            total_distance_m += math.sqrt(dx_m ** 2 + dy_m ** 2)

        return total_distance_m

    def estimate_flight_time(self, drone_speed_mps: float) -> float:
        """
        Estimate the total mission flight time given a constant drone speed.

        Parameters
        ----------
        drone_speed_mps : float   Drone airspeed in metres per second.

        Returns
        -------
        float   Estimated flight duration in seconds.
        """
        if drone_speed_mps <= 0:
            raise ValueError("Drone speed must be positive.")

        return self.compute_route_distance() / drone_speed_mps
