"""
coordinate_converter.py — WGS84 ↔ UTM coordinate system conversion
===================================================================
Converts GPS coordinates to/from UTM using pyproj.
Auto-detects UTM zone from dataset centroid for consistent projection.
"""

from __future__ import annotations

from typing import Sequence
import warnings

try:
    from pyproj import Transformer, CRS
except ImportError:
    raise ImportError(
        "pyproj is required for coordinate conversion. "
        "Install with: pip install pyproj"
    )

import numpy as np


class CoordinateConverter:
    """
    Converts between WGS84 (lat/lon) and UTM coordinate systems.

    Automatically detects UTM zone from dataset centroid to ensure all
    coordinates are projected consistently.
    """

    def __init__(self, lats: Sequence[float], lons: Sequence[float]):
        """
        Initialize converter with dataset bounds.

        Args:
            lats: List of latitude values (degrees)
            lons: List of longitude values (degrees)

        Raises:
            ValueError: If lats/lons are empty or invalid
        """
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)

        if len(lats) == 0 or len(lons) == 0:
            raise ValueError("lats and lons must not be empty")
        if lats.ndim != 1 or lats.shape != lons.shape:
            raise ValueError("lats and lons must be one-dimensional arrays of equal length")
        if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
            raise ValueError("coordinates must be finite")
        if np.any(np.abs(lats) > 90) or np.any(np.abs(lons) > 180):
            raise ValueError("coordinates must be valid WGS84 latitude/longitude")

        # Compute centroid
        centroid_lat = float(np.mean(lats))
        centroid_lon = float(np.mean(lons))

        # Determine UTM zone from centroid
        self.utm_zone = self._compute_utm_zone(centroid_lon)
        self.centroid_lat = centroid_lat
        self.centroid_lon = centroid_lon

        # Create transformer: WGS84 (EPSG:4326) → UTM
        utm_crs = f"+proj=utm +zone={self.utm_zone} +datum=WGS84 +units=m +no_defs"
        if centroid_lat < 0:
            utm_crs += " +south"
        self._crs_wgs84 = CRS.from_epsg(4326)
        self._crs_utm = CRS.from_string(utm_crs)

        self._transformer_to_utm = Transformer.from_crs(
            self._crs_wgs84, self._crs_utm, always_xy=True
        )
        self._transformer_to_wgs84 = Transformer.from_crs(
            self._crs_utm, self._crs_wgs84, always_xy=True
        )

    @staticmethod
    def _compute_utm_zone(lon: float) -> int:
        """
        Compute UTM zone number from longitude (WGS84).

        Args:
            lon: Longitude in degrees (-180 to 180)

        Returns:
            UTM zone number (1–60)
        """
        # UTM zones are 6 degrees wide, starting at −180°
        zone = int((lon + 180.0) / 6.0) + 1
        return max(1, min(60, zone))  # Clamp to [1, 60]

    @property
    def utm_zone_str(self) -> str:
        """Returns UTM zone as formatted string (e.g., '11S')."""
        hemisphere = 'N' if self.centroid_lat >= 0 else 'S'
        return f"{self.utm_zone}{hemisphere}"

    @property
    def crs_wkt(self) -> str:
        """
        Returns CRS in WKT format suitable for rasterio/GDAL.
        """
        return self._crs_utm.to_wkt()

    def to_utm(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Convert single WGS84 coordinate to UTM.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)

        Returns:
            (easting, northing) in metres
        """
        easting, northing = self._transformer_to_utm.transform(lon, lat)
        return float(easting), float(northing)

    def to_wgs84(self, easting: float, northing: float) -> tuple[float, float]:
        """
        Convert single UTM coordinate to WGS84.

        Args:
            easting: Easting in metres
            northing: Northing in metres

        Returns:
            (latitude, longitude) in degrees
        """
        lon, lat = self._transformer_to_wgs84.transform(easting, northing)
        return float(lat), float(lon)

    def batch_to_utm(
        self, lats: Sequence[float], lons: Sequence[float]
    ) -> np.ndarray:
        """
        Convert multiple WGS84 coordinates to UTM (vectorized).

        Args:
            lats: Array-like of latitudes (degrees)
            lons: Array-like of longitudes (degrees)

        Returns:
            (N, 2) array of (easting, northing) in metres
        """
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)

        if lats.shape != lons.shape:
            raise ValueError("lats and lons must have the same shape")

        # Flatten to 1D for transformation
        shape_orig = lats.shape
        lats_flat = lats.ravel()
        lons_flat = lons.ravel()

        # Transform
        eastings, northings = self._transformer_to_utm.transform(
            lons_flat, lats_flat
        )

        # Stack and reshape back
        result = np.column_stack([eastings, northings])
        if shape_orig:
            result = result.reshape((*shape_orig, 2))

        return result

    def batch_to_wgs84(
        self, eastings: Sequence[float], northings: Sequence[float]
    ) -> np.ndarray:
        """
        Convert multiple UTM coordinates to WGS84 (vectorized).

        Args:
            eastings: Array-like of eastings (metres)
            northings: Array-like of northings (metres)

        Returns:
            (N, 2) array of (latitude, longitude) in degrees
        """
        eastings = np.asarray(eastings, dtype=np.float64)
        northings = np.asarray(northings, dtype=np.float64)

        if eastings.shape != northings.shape:
            raise ValueError("eastings and northings must have the same shape")

        # Flatten for transformation
        shape_orig = eastings.shape
        eastings_flat = eastings.ravel()
        northings_flat = northings.ravel()

        # Transform
        lons, lats = self._transformer_to_wgs84.transform(
            eastings_flat, northings_flat
        )

        # Stack and reshape
        result = np.column_stack([lats, lons])
        if shape_orig:
            result = result.reshape((*shape_orig, 2))

        return result
