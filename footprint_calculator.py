"""
footprint_calculator.py  —  Task 1
====================================
Drone Footprint Calculator: converts drone telemetry into the four GPS
corner-coordinates that the camera image covers on the ground.

Math summary
------------
* **Ground sampling distance (GSD)**: At altitude H (metres), and with a
  half-FOV angle α, the half-width of the footprint on a flat plane is
      half_width = H · tan(α/2)          [metres]
  Applied independently for horizontal and vertical FOV.

* **Corner offsets in a local ENU frame**: The four corners are at
      (±half_width_x, ±half_width_y)      [metres, East-North]
  relative to the nadir point (projection of the drone straight down).

* **Metres → degrees conversion**:  Using the WGS-84 approximation
      Δlat [°] = Δnorth [m] / 111_320
      Δlon [°] = Δeast  [m] / (111_320 · cos(lat_rad))
  This is valid for small extents (altitude < a few km, no curvature needed).

* **Heading rotation**: The drone's compass heading θ rotates the local
  East-North offsets before they are added to the nadir GPS position:
      east'  =  east · cos(θ) − north · sin(θ)
      north' =  east · sin(θ) + north · cos(θ)
  where θ is measured clockwise from North and North = 0°.

* **Oblique stubs**: Tilted-gimbal oblique imagery shifts the footprint
  forward/backwards along the flight direction and changes the far/near
  edge differently (trapezoid, not rectangle). This is left as a stub.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models import CameraConfig, DroneTelemetry, GPSCoordinate, GimbalMode


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FootprintResult:
    """
    Four GPS corners of the image footprint, ordered as:
        top-left, top-right, bottom-right, bottom-left
    (when North is "up" and the drone heading is 0°).

    Attributes
    ----------
    top_left     : GPSCoordinate
    top_right    : GPSCoordinate
    bottom_right : GPSCoordinate
    bottom_left  : GPSCoordinate
    nadir        : GPSCoordinate  — the ground point directly below the drone
    ground_width_m  : float  — footprint width  in metres (along drone X-axis)
    ground_height_m : float  — footprint height in metres (along drone Y-axis)
    """
    top_left:     GPSCoordinate
    top_right:    GPSCoordinate
    bottom_right: GPSCoordinate
    bottom_left:  GPSCoordinate
    nadir:        GPSCoordinate
    ground_width_m:  float
    ground_height_m: float

    @property
    def corners(self) -> list[GPSCoordinate]:
        # מחזיר את 4 הפינות בסדר: שמאל-עליון, ימין-עליון, ימין-תחתון, שמאל-תחתון
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    def __repr__(self) -> str:
        return (
            f"FootprintResult(\n"
            f"  TL={self.top_left}\n"
            f"  TR={self.top_right}\n"
            f"  BR={self.bottom_right}\n"
            f"  BL={self.bottom_left}\n"
            f"  width={self.ground_width_m:.1f}m  height={self.ground_height_m:.1f}m\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Calculator class
# ---------------------------------------------------------------------------

class FootprintCalculator:
    """
    Computes the four-corner GPS footprint of a nadir (or oblique) camera
    image captured by a drone.

    Parameters
    ----------
    telemetry : DroneTelemetry
        A complete snapshot of the drone state at shutter time.

    Usage
    -----
    >>> calc   = FootprintCalculator(telemetry)
    >>> result = calc.compute()
    >>> print(result.corners)
    """

    # WGS-84 approximation: metres per degree of latitude (constant globally)
    METRES_PER_LAT_DEG: float = 111_320.0

    def __init__(self, telemetry: DroneTelemetry) -> None:
        self._tel = telemetry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> FootprintResult:
        """
        פונקציה ראשית — מוודאת תקינות קלט, ואז מפנה לחישוב nadir או oblique.
        """
        self._validate()  # בדיקת תקינות קלט לפני כל חישוב

        if self._tel.camera.gimbal_mode is GimbalMode.NADIR:
            return self._compute_nadir()   # מצלמה ישר למטה — מממומש
        elif self._tel.camera.gimbal_mode is GimbalMode.OBLIQUE:
            return self._compute_oblique() # מצלמה מוטה — stub (עתידי)
        else:
            raise ValueError(f"Unknown GimbalMode: {self._tel.camera.gimbal_mode}")

    # ------------------------------------------------------------------
    # Nadir implementation  (Task 1 — FULLY IMPLEMENTED)
    # ------------------------------------------------------------------

    def _compute_nadir(self) -> FootprintResult:
        """
        מחשב את הטביעת רגל כשהמצלמה מכוונת ישר למטה (nadir).
        תוצאה: מלבן GPS של 4 פינות, לאחר רוטציה לפי כיוון העופרת.
        """
        drone_pos   = self._tel.position    # מיקום GPS של הרחפן (כולל גובה)
        altitude_m  = drone_pos.alt         # גובה מעל הקרקע במטרים
        heading_deg = self._tel.heading     # כיוון טיסה בדרגות מהצפון
        camera      = self._tel.camera      # קונפיגורציית המצלמה

        # שלב 1: חצי רוחב וחצי גובה של הטביעה במטרים
        # נוסחה: half = גובה * tan(FOV/2)
        # ככל שהגובה גדול יותר — הטביעה גדולה יותר
        half_w_m = altitude_m * math.tan(math.radians(camera.fov_h / 2.0))
        half_h_m = altitude_m * math.tan(math.radians(camera.fov_v / 2.0))

        # שלב 2: הגדרת 4 הפינות במערכת צירים מקומית (מזרח-צפון, במטרים)
        # לפני רוטציה, "ימין" = מזרח (+x), "קדימה" = צפון (+y)
        raw_corners_en = np.array([
            [-half_w_m,  half_h_m],   # פינה שמאל-עליון  (מערב, צפון)
            [ half_w_m,  half_h_m],   # פינה ימין-עליון  (מזרח, צפון)
            [ half_w_m, -half_h_m],   # פינה ימין-תחתון (מזרח, דרום)
            [-half_w_m, -half_h_m],   # פינה שמאל-תחתון (מערב, דרום)
        ])  # צורה: (4, 2)

        # שלב 3: רוטציה לפי כיוון הרחפן (heading)
        # כיוון נמדד בכיוון השעון מהצפון.
        # רוטציה בכיוון השעון ב-θ: R = [[cos θ, sin θ], [-sin θ, cos θ]]
        # (לא CCW — [[cos θ, -sin θ], [sin θ, cos θ]] — שמניב כיוון שגוי לכל heading ≠ 0°/180°)
        theta = math.radians(heading_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = np.array([
            [ cos_t,  sin_t],
            [-sin_t,  cos_t],
        ])
        rotated_en = (R @ raw_corners_en.T).T  # (4, 2) — כל שורה = (מזרח, צפון) אחרי רוטציה

        # שלב 4: המרת הפרשים ממטרים לדרגות GPS
        # Δlat = Δצפון / 111320
        # Δlon = Δמזרח / (111320 * cos(lat))
        lat_rad = math.radians(drone_pos.lat)
        metres_per_lon_deg = self.METRES_PER_LAT_DEG * math.cos(lat_rad)  # מטרים לדרגה אורך משתנה לפי רוחב

        gps_corners: list[GPSCoordinate] = []
        for east_m, north_m in rotated_en:
            d_lat = north_m / self.METRES_PER_LAT_DEG  # הפרש רוחב בדרגות
            d_lon = east_m  / metres_per_lon_deg        # הפרש אורך בדרגות
            gps_corners.append(
                GPSCoordinate(
                    lat = drone_pos.lat + d_lat,
                    lon = drone_pos.lon + d_lon,
                    alt = 0.0,  # הפינות נמצאות על הקרקע (גובה 0)
                )
            )

        # נקודת ה-nadir — הטיה של הרחפן ישר למטה
        nadir = GPSCoordinate(lat=drone_pos.lat, lon=drone_pos.lon, alt=0.0)

        return FootprintResult(
            top_left     = gps_corners[0],
            top_right    = gps_corners[1],
            bottom_right = gps_corners[2],
            bottom_left  = gps_corners[3],
            nadir        = nadir,
            ground_width_m  = 2.0 * half_w_m,   # רוחב כולל = 2 * חצי רוחב
            ground_height_m = 2.0 * half_h_m,   # גובה כולל = 2 * חצי גובה
        )

    # ------------------------------------------------------------------
    # Oblique implementation  (Task 1)
    # ------------------------------------------------------------------

    def _compute_oblique(self) -> FootprintResult:
        """
        מצלמה מוטה (oblique) יוצרת מלבן הזזה (parallelogram) על הקרקע.

        Convention
        ----------
        gimbal_pitch = 0°  →  nadir (identical to _compute_nadir result)
        gimbal_pitch > 0°  →  forward tilt along the heading direction

        Geometry (before heading rotation, in local ENU frame)
        -------------------------------------------------------
        Along-heading axis (North before rotation):
          near_dist = H × tan(pitch − fov_v/2)   (can be negative = behind nadir)
          far_dist  = H × tan(pitch + fov_v/2)   (always > near_dist)

        Cross-heading axis (East before rotation):
          half_w    = H × tan(fov_h/2)            (same as nadir)

        Raises ValueError if pitch + fov_v/2 ≥ 90° (camera looks past the horizon).
        """
        drone_pos   = self._tel.position
        altitude_m  = drone_pos.alt
        heading_deg = self._tel.heading
        camera      = self._tel.camera
        pitch       = camera.gimbal_pitch       # degrees; 0 = nadir, positive = forward
        half_angle_v = camera.fov_v / 2.0

        if pitch + half_angle_v >= 90.0:
            raise ValueError(
                f"gimbal_pitch ({pitch}°) + fov_v/2 ({half_angle_v}°) >= 90°: "
                "camera looks past or at the horizon — footprint is unbounded."
            )

        # Along-heading distances from the nadir point (metres)
        near_dist_m = altitude_m * math.tan(math.radians(pitch - half_angle_v))
        far_dist_m  = altitude_m * math.tan(math.radians(pitch + half_angle_v))

        # Cross-heading half-width (same as nadir)
        half_w_m = altitude_m * math.tan(math.radians(camera.fov_h / 2.0))

        # Corners in local ENU (East, North) — TL, TR, BR, BL order
        # "top" = far (forward along heading before rotation)
        # "bottom" = near (behind / less forward)
        raw_corners_en = np.array([
            [-half_w_m, far_dist_m ],   # TL: left-far
            [ half_w_m, far_dist_m ],   # TR: right-far
            [ half_w_m, near_dist_m],   # BR: right-near
            [-half_w_m, near_dist_m],   # BL: left-near
        ])

        # Heading rotation (identical to nadir)
        theta = math.radians(heading_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = np.array([
            [ cos_t, -sin_t],
            [ sin_t,  cos_t],
        ])
        rotated_en = (R @ raw_corners_en.T).T

        # ENU metres → GPS degrees
        lat_rad = math.radians(drone_pos.lat)
        metres_per_lon_deg = self.METRES_PER_LAT_DEG * math.cos(lat_rad)

        gps_corners: list[GPSCoordinate] = []
        for east_m, north_m in rotated_en:
            d_lat = north_m / self.METRES_PER_LAT_DEG
            d_lon = east_m  / metres_per_lon_deg
            gps_corners.append(
                GPSCoordinate(lat=drone_pos.lat + d_lat, lon=drone_pos.lon + d_lon, alt=0.0)
            )

        nadir = GPSCoordinate(lat=drone_pos.lat, lon=drone_pos.lon, alt=0.0)

        return FootprintResult(
            top_left     = gps_corners[0],
            top_right    = gps_corners[1],
            bottom_right = gps_corners[2],
            bottom_left  = gps_corners[3],
            nadir        = nadir,
            ground_width_m  = 2.0 * half_w_m,
            ground_height_m = far_dist_m - near_dist_m,  # along-heading extent
        )

    def _validate(self) -> None:
        """בודקת שהגובה חיובי ושה-FOV בטווח הגיוני — לפני כל חישוב."""
        alt = self._tel.position.alt
        if alt <= 0:
            raise ValueError(f"Altitude must be positive, got {alt} m.")

        cam = self._tel.camera
        for fov_name, fov_val in [("fov_h", cam.fov_h), ("fov_v", cam.fov_v)]:
            if not (0 < fov_val < 180):
                raise ValueError(
                    f"Camera {fov_name} must be in (0°, 180°), got {fov_val}°."
                )


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly to verify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models import CameraConfig, DroneTelemetry, GPSCoordinate, GimbalMode

    # Example: drone hovering at 100 m above Tel Aviv city centre,
    # heading North (0°), 90°×60° FOV camera in nadir mode.
    telemetry = DroneTelemetry(
        position=GPSCoordinate(lat=32.0853, lon=34.7818, alt=100.0),
        heading=0.0,
        camera=CameraConfig(fov_h=90.0, fov_v=60.0, gimbal_mode=GimbalMode.NADIR),
    )

    calc   = FootprintCalculator(telemetry)
    result = calc.compute()

    print("=== Footprint Calculator — Nadir Example ===")
    print(result)
    print(f"\nGround coverage: {result.ground_width_m:.1f} m × {result.ground_height_m:.1f} m")
