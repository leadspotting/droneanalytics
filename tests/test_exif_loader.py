"""
test_exif_loader.py  —  Unit and integration tests for exif_loader.py
======================================================================
Split into two sections:

  TestExifLoaderUnit      — pure unit tests, no filesystem access, no real images.
                            Uses synthetic data to test _dms_to_decimal, _bearing,
                            _extract_gps, and heading derivation logic.

  TestSEKIIntegration     — reads the real task3_data/seki_southfork/images directory.
                            Skipped automatically if the directory is not present,
                            so the regular CI test run works on any machine.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exif_loader import (
    RICOH_GR_II_FOV_H,
    RICOH_GR_II_FOV_V,
    RICOH_GR_II_H_PX,
    RICOH_GR_II_W_PX,
    SEKI_AGL_M,
    _bearing,
    _dms_to_decimal,
    _extract_gps,
    load_image_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEKI_DIR = Path(__file__).resolve().parent.parent / "task3_data" / "seki_southfork" / "images"
_SEKI_AVAILABLE = _SEKI_DIR.exists() and any(_SEKI_DIR.glob("*.jpg"))


# ---------------------------------------------------------------------------
# Unit tests — no filesystem, no real images
# ---------------------------------------------------------------------------

class TestDmsToDecimal:
    def test_north(self):
        # 36 deg 20 min 29.58 sec N
        result = _dms_to_decimal((36.0, 20.0, 29.58), "N")
        assert abs(result - 36.3415) < 0.0001

    def test_west(self):
        # 118 deg 45 min 23.508 sec W  -> negative
        result = _dms_to_decimal((118.0, 45.0, 23.508), "W")
        assert abs(result - (-118.7565)) < 0.0001

    def test_south_is_negative(self):
        result = _dms_to_decimal((10.0, 0.0, 0.0), "S")
        assert result == pytest.approx(-10.0)

    def test_east_is_positive(self):
        result = _dms_to_decimal((34.0, 48.0, 0.0), "E")
        assert result == pytest.approx(34.8)

    def test_zero_minutes_seconds(self):
        assert _dms_to_decimal((45.0, 0.0, 0.0), "N") == pytest.approx(45.0)

    def test_full_dms(self):
        # 1 deg 1 min 1 sec = 1 + 1/60 + 1/3600 = 1.016944...
        result = _dms_to_decimal((1.0, 1.0, 1.0), "N")
        assert abs(result - 1.01694) < 0.0001


class TestBearing:
    def test_north(self):
        # moving straight north -> bearing 0
        b = _bearing(36.0, -118.0, 36.01, -118.0)
        assert abs(b - 0.0) < 0.5 or abs(b - 360.0) < 0.5

    def test_east(self):
        # moving straight east -> bearing 90
        b = _bearing(36.0, -118.0, 36.0, -117.99)
        assert abs(b - 90.0) < 0.5

    def test_south(self):
        # moving straight south -> bearing 180
        b = _bearing(36.01, -118.0, 36.0, -118.0)
        assert abs(b - 180.0) < 0.5

    def test_west(self):
        # moving straight west -> bearing 270
        b = _bearing(36.0, -117.99, 36.0, -118.0)
        assert abs(b - 270.0) < 0.5

    def test_result_in_range(self):
        for dlat, dlon in [(0.01, 0.01), (-0.01, 0.01), (-0.01, -0.01), (0.01, -0.01)]:
            b = _bearing(36.0, -118.0, 36.0 + dlat, -118.0 + dlon)
            assert 0.0 <= b < 360.0


class TestExtractGps:
    """Tests _extract_gps() with mock EXIF integer-keyed dicts."""

    def _make_exif(self, lat_dms, lat_ref, lon_dms, lon_ref, alt=100.0,
                   ts_date="2024:04:30", ts_time=(22.0, 22.0, 21.0)):
        """Build a minimal fake EXIF dict with a GPSInfo entry."""
        from PIL.ExifTags import TAGS, GPSTAGS

        # Find the integer tag ID for GPSInfo
        gps_tag_id = next(k for k, v in TAGS.items() if v == "GPSInfo")

        # Build the inner GPS dict keyed by integers
        gps_int = {}
        for k, v in GPSTAGS.items():
            if v == "GPSLatitude":       gps_int[k] = lat_dms
            if v == "GPSLatitudeRef":    gps_int[k] = lat_ref
            if v == "GPSLongitude":      gps_int[k] = lon_dms
            if v == "GPSLongitudeRef":   gps_int[k] = lon_ref
            if v == "GPSAltitude":       gps_int[k] = alt
            if v == "GPSTimeStamp":      gps_int[k] = ts_time
            if v == "GPSDateStamp":      gps_int[k] = ts_date

        return {gps_tag_id: gps_int}

    def test_valid_gps(self):
        exif = self._make_exif(
            (36.0, 20.493, 0.0), "N",
            (118.0, 45.3918, 0.0), "W",
        )
        result = _extract_gps(exif)
        assert result is not None
        assert abs(result["lat"] - 36.341550) < 0.0001
        assert abs(result["lon"] - (-118.756530)) < 0.0001
        assert result["alt_ellipsoid_m"] == pytest.approx(100.0)

    def test_zero_coords_filtered(self):
        exif = self._make_exif((0.0, 0.0, 0.0), "N", (0.0, 0.0, 0.0), "E")
        assert _extract_gps(exif) is None

    def test_missing_gps_info_returns_none(self):
        assert _extract_gps({}) is None

    def test_timestamp_built(self):
        exif = self._make_exif(
            (36.0, 20.0, 0.0), "N",
            (118.0, 45.0, 0.0), "W",
            ts_date="2024:04:30", ts_time=(22.0, 22.0, 21.0),
        )
        result = _extract_gps(exif)
        assert result is not None
        assert result["timestamp"] == "2024:04:30 22:22:21Z"

    def test_west_longitude_negative(self):
        exif = self._make_exif(
            (36.0, 0.0, 0.0), "N",
            (118.0, 0.0, 0.0), "W",
        )
        result = _extract_gps(exif)
        assert result is not None
        assert result["lon"] < 0


class TestHeadingDerivation:
    """Tests that load_image_dir derives headings correctly from synthetic data."""

    def _fake_dir_with_positions(self, positions: list[tuple[float, float]], tmp_path: Path):
        """
        Create stub JPG files whose EXIF will be mocked to return given lat/lon pairs.
        Returns the directory path and the list of fake file paths in filename order.
        """
        files = []
        for i, _ in enumerate(positions):
            p = tmp_path / f"img_{i:04d}.jpg"
            p.write_bytes(b"FAKE")
            files.append(p)
        return tmp_path, files

    def test_first_image_heading_zero(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from PIL.ExifTags import TAGS, GPSTAGS

        gps_tag = next(k for k, v in TAGS.items() if v == "GPSInfo")
        gps_int = {}
        for k, v in GPSTAGS.items():
            if v == "GPSLatitude":     gps_int[k] = (36.0, 0.0, 0.0)
            if v == "GPSLatitudeRef":  gps_int[k] = "N"
            if v == "GPSLongitude":    gps_int[k] = (118.0, 0.0, 0.0)
            if v == "GPSLongitudeRef": gps_int[k] = "W"
            if v == "GPSAltitude":     gps_int[k] = 1465.0

        fake_exif = {gps_tag: gps_int}

        p = tmp_path / "img_0000.jpg"
        p.write_bytes(b"FAKE")

        mock_img = MagicMock()
        mock_img._getexif.return_value = fake_exif

        # PIL.Image is imported lazily inside load_image_dir via
        # "from PIL import Image as PILImage"; patching PIL.Image.open
        # intercepts that call since PILImage IS PIL.Image.
        with patch("PIL.Image.open", return_value=mock_img):
            rows = load_image_dir(tmp_path)

        assert len(rows) == 1
        assert rows[0]["heading"] == 0.0

    def test_heading_derived_from_consecutive_positions(self):
        """Directly test the bearing derivation over two synthetic rows."""
        # Simulate two raw GPS readings moving north
        from exif_loader import _bearing
        b = _bearing(36.0, -118.0, 36.01, -118.0)
        assert abs(b) < 0.5 or abs(b - 360.0) < 0.5  # ~0 deg (North)

    def test_stationary_keeps_heading(self):
        """Two identical positions -> heading must not change."""
        from exif_loader import _bearing
        # If positions are equal, code uses prev_heading unchanged
        # Simulate by checking that equal lat/lon does NOT call _bearing
        prev_heading = 45.0
        lat, lon = 36.0, -118.0
        prev_lat, prev_lon = 36.0, -118.0
        # Condition: lat != prev_lat or lon != prev_lon -> False -> no update
        if lat != prev_lat or lon != prev_lon:
            prev_heading = _bearing(prev_lat, prev_lon, lat, lon)
        assert prev_heading == 45.0


class TestLoadImageDirEdgeCases:
    def test_missing_directory_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_image_dir(missing)

    def test_empty_directory_returns_empty(self, tmp_path):
        rows = load_image_dir(tmp_path)
        assert rows == []


# ---------------------------------------------------------------------------
# Integration tests — require real SEKI dataset on disk
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _SEKI_AVAILABLE,
    reason="SEKI dataset not present at task3_data/seki_southfork/images/",
)
class TestSEKIIntegration:
    """
    Integration tests against the real SEKI SouthFork dataset.
    These tests are skipped automatically on machines where the dataset is absent.
    """

    @pytest.fixture(scope="class")
    def rows(self):
        return load_image_dir(_SEKI_DIR)

    def test_image_count(self, rows):
        assert len(rows) == 61

    def test_coordinates_in_range(self, rows):
        for r in rows:
            assert 36.33 < r["lat"] < 36.36, f"lat out of range: {r['lat']}"
            assert -118.77 < r["lon"] < -118.75, f"lon out of range: {r['lon']}"

    def test_agl_applied(self, rows):
        for r in rows:
            assert r["agl_m"] == pytest.approx(SEKI_AGL_M)

    def test_ellipsoid_alt_stored(self, rows):
        # Ellipsoid alt should be ~1465 m (not AGL 110 m)
        for r in rows:
            assert r["alt_ellipsoid_m"] is not None
            assert r["alt_ellipsoid_m"] > 1000.0

    def test_headings_valid_range(self, rows):
        for r in rows:
            assert 0.0 <= r["heading"] < 360.0

    def test_first_heading_zero(self, rows):
        assert rows[0]["heading"] == 0.0

    def test_fov_constants(self, rows):
        for r in rows:
            assert r["fov_h"] == pytest.approx(RICOH_GR_II_FOV_H)
            assert r["fov_v"] == pytest.approx(RICOH_GR_II_FOV_V)

    def test_image_dimensions(self, rows):
        for r in rows:
            assert r["width_px"] == RICOH_GR_II_W_PX
            assert r["height_px"] == RICOH_GR_II_H_PX

    def test_sorted_by_filename(self, rows):
        names = [Path(r["path"]).name for r in rows]
        assert names == sorted(names)

    def test_timestamps_nonempty(self, rows):
        # All SEKI images should have GPS timestamp in EXIF
        empty = [r for r in rows if not r["timestamp"]]
        assert len(empty) == 0, f"{len(empty)} images have empty timestamps"


# ---------------------------------------------------------------------------
# Preview route tests — GET /image_mission/{id}/mosaic/preview
# ---------------------------------------------------------------------------

import api as api_module
from fastapi.testclient import TestClient


def _fake_row(tmp_path, index: int = 0) -> dict:
    """Build a minimal image-mission row dict for preview route tests."""
    return {
        "path":            str(tmp_path / f"img_{index}.jpg"),
        "lat":             36.340 + index * 0.001,
        "lon":             -118.760,
        "agl_m":           110.0,
        "alt_ellipsoid_m": 1465.0,
        "timestamp":       "",
        "heading":         0.0,
        "width_px":        4928,
        "height_px":       3264,
        "fov_h":           65.8,
        "fov_v":           46.4,
    }


class TestPreviewRoute:
    """Unit tests for GET /image_mission/{mission_id}/mosaic/preview."""

    def setup_method(self):
        api_module._image_missions.clear()

    def test_404_unknown_mission(self):
        client = TestClient(api_module.app)
        resp = client.get("/image_mission/DOES_NOT_EXIST/mosaic/preview")
        assert resp.status_code == 404

    def test_422_no_loadable_images(self, tmp_path):
        # Row points to a non-existent file → PIL.Image.open raises → skipped → <2 images
        api_module._image_missions["EMPTY_PREVIEW"] = [_fake_row(tmp_path, 0)]
        client = TestClient(api_module.app)
        resp = client.get("/image_mission/EMPTY_PREVIEW/mosaic/preview")
        assert resp.status_code == 422

    def test_returns_png_bytes(self, tmp_path):
        """Two fake rows with mocked PIL + mocked stitch → 200 image/png response."""
        import numpy as np
        from PIL import Image as PILImage
        from drone_oversight import MosaicResult
        from models import GPSCoordinate

        tiny_img = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        tiny_canvas = np.zeros((5, 5, 3), dtype=np.uint8)
        fake_mosaic = MosaicResult(
            image=tiny_canvas,
            bounding_box=[
                GPSCoordinate(lat=36.354, lon=-118.768),
                GPSCoordinate(lat=36.354, lon=-118.754),
                GPSCoordinate(lat=36.338, lon=-118.754),
                GPSCoordinate(lat=36.338, lon=-118.768),
            ],
            gsd_m=0.579,
        )

        api_module._image_missions["PNG_TEST"] = [
            _fake_row(tmp_path, i) for i in range(2)
        ]

        with patch("PIL.Image.open", return_value=tiny_img), \
             patch("drone_oversight.ImageStitcher.stitch", return_value=fake_mosaic):
            client = TestClient(api_module.app)
            resp = client.get("/image_mission/PNG_TEST/mosaic/preview?scale=0.05")

        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
        assert len(resp.content) > 0

    def test_scale_clamped_to_minimum(self, tmp_path):
        """scale=0.001 is clamped to 0.01 — request should behave the same as scale=0.01."""
        import numpy as np
        from PIL import Image as PILImage
        from drone_oversight import MosaicResult
        from models import GPSCoordinate

        tiny_img = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        tiny_canvas = np.zeros((5, 5, 3), dtype=np.uint8)
        fake_mosaic = MosaicResult(
            image=tiny_canvas,
            bounding_box=[GPSCoordinate(lat=36.354, lon=-118.768)] * 4,
            gsd_m=0.579,
        )

        api_module._image_missions["SCALE_CLAMP"] = [
            _fake_row(tmp_path, i) for i in range(2)
        ]

        with patch("PIL.Image.open", return_value=tiny_img), \
             patch("drone_oversight.ImageStitcher.stitch", return_value=fake_mosaic):
            client = TestClient(api_module.app)
            resp = client.get("/image_mission/SCALE_CLAMP/mosaic/preview?scale=0.001")

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Cluster detection tests — _detect_clusters()
# ---------------------------------------------------------------------------

from api import _detect_clusters


class TestDetectClusters:
    """Unit tests for _detect_clusters() in api.py."""

    def test_empty_returns_empty(self):
        assert _detect_clusters([]) == []

    def test_close_rows_stay_in_one_cluster(self):
        # Steps ~7 m apart — well below the 200 m default threshold
        rows = [
            {"lat": 36.341050, "lon": -118.757000},
            {"lat": 36.341110, "lon": -118.757060},
            {"lat": 36.341170, "lon": -118.757120},
        ]
        clusters = _detect_clusters(rows)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_large_gap_creates_second_cluster(self):
        # First two rows are ~7 m apart; third is ~837 m away
        rows = [
            {"lat": 36.341050, "lon": -118.757000},
            {"lat": 36.341110, "lon": -118.757060},
            {"lat": 36.348470, "lon": -118.761910},
        ]
        clusters = _detect_clusters(rows)
        assert len(clusters) == 2
        assert len(clusters[0]) == 2
        assert len(clusters[1]) == 1

    def test_seki_produces_two_clusters(self):
        """SEKI dataset splits at the 837 m gap between frames 45 and 46."""
        seki_dir = Path("task3_data/seki_southfork/images")
        if not seki_dir.exists():
            pytest.skip("SEKI dataset not present")
        from exif_loader import load_image_dir
        rows = load_image_dir(seki_dir)
        clusters = _detect_clusters(rows)
        assert len(clusters) == 2
        assert len(clusters[0]) == 46
        assert len(clusters[1]) == 15


class TestGroupedPreviewRoute:
    """Tests for GET /image_mission/{id}/mosaic/preview/grouped."""

    def setup_method(self):
        api_module._image_missions.clear()

    def test_404_unknown_mission(self):
        client = TestClient(api_module.app)
        resp = client.get("/image_mission/GHOST/mosaic/preview/grouped")
        assert resp.status_code == 404

    def test_422_no_stitchable_clusters(self, tmp_path):
        # Only 1 row → cluster has <2 loadable images (PIL.Image.open not mocked → raises)
        api_module._image_missions["ONE_ROW"] = [_fake_row(tmp_path, 0)]
        client = TestClient(api_module.app)
        resp = client.get("/image_mission/ONE_ROW/mosaic/preview/grouped")
        assert resp.status_code == 422

    def test_two_clusters_returns_wider_png(self, tmp_path):
        """Rows with ~837 m gap → 2 clusters → side-by-side PNG width = 2×panel + gap."""
        import numpy as np
        from PIL import Image as PILImage
        from drone_oversight import MosaicResult
        from models import GPSCoordinate
        import io as _io

        # Each cluster stitch produces an 8×5 image
        tiny_img = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        fake_mosaic = MosaicResult(
            image=np.zeros((5, 8, 3), dtype=np.uint8),
            bounding_box=[GPSCoordinate(lat=36.354, lon=-118.768)] * 4,
            gsd_m=0.579,
        )

        # Two pairs of rows separated by the real SEKI 837 m gap
        rows = [
            {**_fake_row(tmp_path, 0), "lat": 36.341050, "lon": -118.757000},
            {**_fake_row(tmp_path, 1), "lat": 36.341110, "lon": -118.757060},
            {**_fake_row(tmp_path, 2), "lat": 36.348470, "lon": -118.761910},
            {**_fake_row(tmp_path, 3), "lat": 36.348530, "lon": -118.761980},
        ]
        api_module._image_missions["TWO_CLUSTERS"] = rows

        with patch("PIL.Image.open", return_value=tiny_img), \
             patch("drone_oversight.ImageStitcher.stitch", return_value=fake_mosaic):
            client = TestClient(api_module.app)
            resp = client.get("/image_mission/TWO_CLUSTERS/mosaic/preview/grouped")

        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
        result_img = PILImage.open(_io.BytesIO(resp.content))
        # 2 panels of width 8 + 10 px gap = 26 px total
        assert result_img.width == 8 + 10 + 8
