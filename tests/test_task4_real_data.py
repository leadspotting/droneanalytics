"""
test_task4_real_data.py  -  Strong real-data validation for Task 4
==================================================================

Validates GET /image_mission/{id}/project against independent reference data:

1. RTK GNSS CSV points, used to verify the pure-Python UTM conversion helpers.
2. GeoTIFF georeferencing tags, used as an independent pixel->world reference.

Required files (tests skip automatically if absent):
  task4_data/orthomosaic/SEKI_SouthFork_DebrisFlow_2024-04-30_orthomosaic_3cm.tif
  task4_data/orthomosaic/SEKI_SouthFork_DebrisFlow_2024-04-30_orthomosaic_metadata.xml
  task4_data/gcp_gnss/SEKI_SouthFork_DebrisFlow_2024_RTK_GNSS.csv

Run with:
    python -m pytest test_task4_real_data.py -v
"""

from __future__ import annotations

import csv
import math
import pathlib
import unittest
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
from PIL import Image

import api as api_module


_TASK4_DIR = pathlib.Path("task4_data")
_TIF_FILE = _TASK4_DIR / "orthomosaic" / (
    "SEKI_SouthFork_DebrisFlow_2024-04-30_orthomosaic_3cm.tif"
)
_XML_FILE = _TASK4_DIR / "orthomosaic" / (
    "SEKI_SouthFork_DebrisFlow_2024-04-30_orthomosaic_metadata.xml"
)
_CSV_FILE = _TASK4_DIR / "gcp_gnss" / (
    "SEKI_SouthFork_DebrisFlow_2024_RTK_GNSS.csv"
)

_TASK4_AVAILABLE = _TIF_FILE.exists() and _XML_FILE.exists() and _CSV_FILE.exists()
_SKIP_REASON = "task4_data reference files not present"

_MISSION_ID = "TASK4_REAL_VALIDATION"

# WGS84 UTM constants. NAD83(2011) differs only at the cm level for this test.
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)
_WGS84_EP2 = _WGS84_E2 / (1 - _WGS84_E2)
_UTM_K0 = 0.9996
_UTM_ZONE = 11


def _load_ortho_bbox(xml_path: pathlib.Path) -> dict:
    root = ET.parse(xml_path).getroot()
    b = root.find(".//bounding")
    return {
        "west": float(b.find("westbc").text),
        "east": float(b.find("eastbc").text),
        "north": float(b.find("northbc").text),
        "south": float(b.find("southbc").text),
    }


def _load_geotiff_reference(tif_path: pathlib.Path) -> dict:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(tif_path) as img:
        scale = img.tag_v2.get(33550)
        tie = img.tag_v2.get(33922)
        if scale is None or tie is None:
            raise RuntimeError("GeoTIFF missing georeferencing tags 33550/33922.")
        return {
            "width_px": img.size[0],
            "height_px": img.size[1],
            "scale_x_m": float(scale[0]),
            "scale_y_m": float(scale[1]),
            "ul_easting_m": float(tie[3]),
            "ul_northing_m": float(tie[4]),
        }


def _geotiff_latlon_bbox(ref: dict) -> dict:
    corners = [
        _utm_to_latlon(*_geotiff_pixel_to_utm(ref, 0, 0)),
        _utm_to_latlon(*_geotiff_pixel_to_utm(ref, ref["width_px"] - 1, 0)),
        _utm_to_latlon(*_geotiff_pixel_to_utm(ref, ref["width_px"] - 1, ref["height_px"] - 1)),
        _utm_to_latlon(*_geotiff_pixel_to_utm(ref, 0, ref["height_px"] - 1)),
    ]
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    return {
        "north": max(lats),
        "south": min(lats),
        "west": min(lons),
        "east": max(lons),
    }


def _load_gcp_points(csv_path: pathlib.Path, max_points: int = 8) -> list[dict]:
    points: list[dict] = []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("observation_time_utc", "").strip():
                points.append({
                    "pid": row["pid"],
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "easting_m": float(row["easting_m"]),
                    "northing_m": float(row["northing_m"]),
                })
            if len(points) >= max_points:
                break
    return points


def _utm_central_meridian_rad(zone: int) -> float:
    return math.radians(zone * 6 - 183)


def _latlon_to_utm(lat_deg: float, lon_deg: float, zone: int = _UTM_ZONE) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = _utm_central_meridian_rad(zone)

    n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = _WGS84_EP2 * math.cos(lat) ** 2
    a = math.cos(lat) * (lon - lon0)

    m = _WGS84_A * (
        (1 - _WGS84_E2 / 4 - 3 * _WGS84_E2 ** 2 / 64 - 5 * _WGS84_E2 ** 3 / 256) * lat
        - (3 * _WGS84_E2 / 8 + 3 * _WGS84_E2 ** 2 / 32 + 45 * _WGS84_E2 ** 3 / 1024) * math.sin(2 * lat)
        + (15 * _WGS84_E2 ** 2 / 256 + 45 * _WGS84_E2 ** 3 / 1024) * math.sin(4 * lat)
        - (35 * _WGS84_E2 ** 3 / 3072) * math.sin(6 * lat)
    )

    easting = _UTM_K0 * n * (
        a
        + (1 - t + c) * a ** 3 / 6
        + (5 - 18 * t + t ** 2 + 72 * c - 58 * _WGS84_EP2) * a ** 5 / 120
    ) + 500000.0

    northing = _UTM_K0 * (
        m
        + n * math.tan(lat) * (
            a ** 2 / 2
            + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
            + (61 - 58 * t + t ** 2 + 600 * c - 330 * _WGS84_EP2) * a ** 6 / 720
        )
    )
    return easting, northing


def _utm_to_latlon(easting_m: float, northing_m: float, zone: int = _UTM_ZONE) -> tuple[float, float]:
    x = easting_m - 500000.0
    m = northing_m / _UTM_K0
    mu = m / (
        _WGS84_A
        * (1 - _WGS84_E2 / 4 - 3 * _WGS84_E2 ** 2 / 64 - 5 * _WGS84_E2 ** 3 / 256)
    )

    e1 = (1 - math.sqrt(1 - _WGS84_E2)) / (1 + math.sqrt(1 - _WGS84_E2))
    j1 = 3 * e1 / 2 - 27 * e1 ** 3 / 32
    j2 = 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32
    j3 = 151 * e1 ** 3 / 96
    j4 = 1097 * e1 ** 4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = _WGS84_EP2 * cos_fp ** 2
    t1 = tan_fp ** 2
    n1 = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_fp ** 2)
    r1 = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * sin_fp ** 2) ** 1.5
    d = x / (n1 * _UTM_K0)

    lat = fp - (n1 * tan_fp / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _WGS84_EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _WGS84_EP2 - 3 * c1 ** 2) * d ** 6 / 720
    )

    lon = _utm_central_meridian_rad(zone) + (
        d
        - (1 + 2 * t1 + c1) * d ** 3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _WGS84_EP2 + 24 * t1 ** 2) * d ** 5 / 120
    ) / cos_fp

    return math.degrees(lat), math.degrees(lon)


def _metres_to_lat_deg(distance_m: float) -> float:
    return distance_m / 111_320.0


def _metres_to_lon_deg(distance_m: float, lat_deg: float) -> float:
    return distance_m / (111_320.0 * math.cos(math.radians(lat_deg)))


def _build_synthetic_rows(center_a: tuple[float, float], center_b: tuple[float, float]) -> list[dict]:
    def _row(path: str, lat: float, lon: float) -> dict:
        return {
            "path": path,
            "lat": lat,
            "lon": lon,
            "agl_m": 110.0,
            "alt_ellipsoid_m": 1465.0,
            "timestamp": "2024-04-30T15:00:00Z",
            "heading": 0.0,
            "width_px": 4928,
            "height_px": 3264,
            "fov_h": 65.8,
            "fov_v": 46.4,
        }

    return [
        _row("/synthetic/A.JPG", center_a[0], center_a[1]),
        _row("/synthetic/B.JPG", center_b[0], center_b[1]),
    ]


def _geotiff_pixel_to_utm(ref: dict, x_px: int, y_px: int) -> tuple[float, float]:
    return (
        ref["ul_easting_m"] + x_px * ref["scale_x_m"],
        ref["ul_northing_m"] - y_px * ref["scale_y_m"],
    )


def _normalised_pixel(src_w: int, src_h: int, dst_w: int, dst_h: int, x: int, y: int) -> tuple[int, int]:
    x_norm = x / max(src_w - 1, 1)
    y_norm = y / max(src_h - 1, 1)
    return (
        round(x_norm * max(dst_w - 1, 1)),
        round(y_norm * max(dst_h - 1, 1)),
    )


@unittest.skipUnless(_TASK4_AVAILABLE, _SKIP_REASON)
class TestTask4RealDataValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ortho_bbox = _load_ortho_bbox(_XML_FILE)
        cls._ortho_ref = _load_geotiff_reference(_TIF_FILE)
        cls._ortho_tif_bbox = _geotiff_latlon_bbox(cls._ortho_ref)
        cls._gcp_points = _load_gcp_points(_CSV_FILE)

        # Match the synthetic mosaic to the GeoTIFF-derived north-up envelope.
        center_lat = (cls._ortho_tif_bbox["north"] + cls._ortho_tif_bbox["south"]) / 2.0
        half_width_deg = _metres_to_lon_deg(142.324 / 2.0, center_lat)
        half_height_deg = _metres_to_lat_deg(94.079 / 2.0)
        cls._rows = _build_synthetic_rows(
            center_a=(
                cls._ortho_tif_bbox["north"] - half_height_deg,
                cls._ortho_tif_bbox["west"] + half_width_deg,
            ),
            center_b=(
                cls._ortho_tif_bbox["south"] + half_height_deg,
                cls._ortho_tif_bbox["east"] - half_width_deg,
            ),
        )

    def setUp(self):
        self.client = TestClient(api_module.app)
        self._saved = api_module._image_missions.copy()
        api_module._image_missions[_MISSION_ID] = self.__class__._rows

        mosaic = self.client.get(f"/image_mission/{_MISSION_ID}/mosaic")
        self.assertEqual(mosaic.status_code, 200)
        data = mosaic.json()
        self._W = data["canvas_width_px"]
        self._H = data["canvas_height_px"]
        self._bb = data["bounding_box"]
        self._lat_min = self._bb["BR"]["lat"]
        self._lat_max = self._bb["TL"]["lat"]
        self._lon_min = self._bb["TL"]["lon"]
        self._lon_max = self._bb["BR"]["lon"]

    def tearDown(self):
        api_module._image_missions.clear()
        api_module._image_missions.update(self._saved)
    def test_pixel_0_0_maps_to_canvas_tl(self):
        data = self.client.get(f"/image_mission/{_MISSION_ID}/project?x=0&y=0").json()
        self.assertAlmostEqual(data["lat"], self._bb["TL"]["lat"], places=5)
        self.assertAlmostEqual(data["lon"], self._bb["TL"]["lon"], places=5)

    def test_pixel_wm1_hm1_maps_to_canvas_br(self):
        data = self.client.get(
            f"/image_mission/{_MISSION_ID}/project?x={self._W - 1}&y={self._H - 1}"
        ).json()
        self.assertAlmostEqual(data["lat"], self._bb["BR"]["lat"], places=5)
        self.assertAlmostEqual(data["lon"], self._bb["BR"]["lon"], places=5)

    def test_utm_conversion_matches_rtk_csv(self):
        tol_m = 0.02
        for point in self.__class__._gcp_points:
            east, north = _latlon_to_utm(point["lat"], point["lon"])
            self.assertAlmostEqual(east, point["easting_m"], delta=tol_m)
            self.assertAlmostEqual(north, point["northing_m"], delta=tol_m)

            lat, lon = _utm_to_latlon(point["easting_m"], point["northing_m"])
            lat_err_m = abs(lat - point["lat"]) * 111_320.0
            lon_err_m = abs(lon - point["lon"]) * 111_320.0 * math.cos(math.radians(point["lat"]))
            self.assertLess(lat_err_m, tol_m)
            self.assertLess(lon_err_m, tol_m)

    def test_centre_pixel_returns_finite_gps_in_range(self):
        data = self.client.get(
            f"/image_mission/{_MISSION_ID}/project?x={self._W // 2}&y={self._H // 2}"
        ).json()
        self.assertTrue(math.isfinite(data["lat"]))
        self.assertTrue(math.isfinite(data["lon"]))
        self.assertGreaterEqual(data["lat"], -90.0)
        self.assertLessEqual(data["lat"], 90.0)
        self.assertGreaterEqual(data["lon"], -180.0)
        self.assertLessEqual(data["lon"], 180.0)

    def test_ortho_bbox_is_plausible_reference(self):
        bb = self.__class__._ortho_bbox
        self.assertGreater(bb["north"], bb["south"])
        self.assertGreater(bb["east"], bb["west"])
        self.assertGreater(bb["south"], 36.0)
        self.assertLess(bb["north"], 37.0)
        self.assertLess(bb["west"], -118.0)
        self.assertGreater(bb["east"], -120.0)

    def test_canvas_bbox_lies_close_to_ortho_extent(self):
        ob = self.__class__._ortho_tif_bbox
        tol_deg = 1e-4
        self.assertAlmostEqual(self._lat_max, ob["north"], delta=tol_deg)
        self.assertAlmostEqual(self._lat_min, ob["south"], delta=tol_deg)
        self.assertAlmostEqual(self._lon_min, ob["west"], delta=tol_deg)
        self.assertAlmostEqual(self._lon_max, ob["east"], delta=tol_deg)

    def test_normalised_geotiff_positions_match_route_outputs(self):
        """
        Compare the API route to the independent GeoTIFF pixel->UTM transform.

        Tolerance is 220 m because the route models a GPS-footprint mosaic, not
        the high-quality SfM orthomosaic itself. This is strong enough to catch
        axis flips, bbox inversion, or gross scaling errors while respecting the
        intended coarse nature of Task 4.
        """
        samples = [
            (0, 0),
            (self._W - 1, 0),
            (0, self._H - 1),
            (self._W - 1, self._H - 1),
            (self._W // 2, self._H // 2),
        ]
        tol_m = 220.0
        ref = self.__class__._ortho_ref

        for x_route, y_route in samples:
            x_tif, y_tif = _normalised_pixel(
                self._W, self._H,
                ref["width_px"], ref["height_px"],
                x_route, y_route,
            )
            east_ref, north_ref = _geotiff_pixel_to_utm(ref, x_tif, y_tif)
            data = self.client.get(
                f"/image_mission/{_MISSION_ID}/project?x={x_route}&y={y_route}"
            ).json()
            east_api, north_api = _latlon_to_utm(data["lat"], data["lon"])
            err_m = math.hypot(east_api - east_ref, north_api - north_ref)
            self.assertLess(err_m, tol_m, msg=f"pixel ({x_route},{y_route}) err={err_m:.2f} m")


if __name__ == "__main__":
    unittest.main(verbosity=2)
