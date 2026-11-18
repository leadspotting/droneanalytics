"""
test_altitude_scaler.py  —  Unit tests for Task 5: AltitudeScaler
==================================================================
Run with:
    python -m pytest test_altitude_scaler.py -v
"""

import math
import unittest

from models import CameraConfig, GimbalMode
from drone_oversight import AltitudeScaler


def _make_camera(fov_h: float = 90.0, fov_v: float = 60.0) -> CameraConfig:
    return CameraConfig(fov_h=fov_h, fov_v=fov_v, gimbal_mode=GimbalMode.NADIR)


class TestComputeGSD(unittest.TestCase):

    def setUp(self):
        self.camera = _make_camera(fov_h=90.0)
        self.scaler = AltitudeScaler(
            reference_altitude_m=100.0,
            reference_gsd_m=0.2,
            camera_config=self.camera,
        )

    def test_gsd_formula_nadir_90fov(self):
        # fov_h=90° → half-angle=45° → ground_width = 2 * H * tan(45°) = 2*H
        # At H=100m, W=1000px: GSD = 200m / 1000 = 0.2 m/px
        gsd = self.scaler.compute_gsd(altitude_m=100.0, image_width_px=1000)
        self.assertAlmostEqual(gsd, 0.2, places=6)

    def test_gsd_proportional_to_altitude(self):
        gsd_100 = self.scaler.compute_gsd(altitude_m=100.0, image_width_px=1000)
        gsd_200 = self.scaler.compute_gsd(altitude_m=200.0, image_width_px=1000)
        self.assertAlmostEqual(gsd_200 / gsd_100, 2.0, places=6)

    def test_gsd_inversely_proportional_to_image_width(self):
        gsd_1000 = self.scaler.compute_gsd(altitude_m=100.0, image_width_px=1000)
        gsd_2000 = self.scaler.compute_gsd(altitude_m=100.0, image_width_px=2000)
        self.assertAlmostEqual(gsd_2000, gsd_1000 / 2.0, places=6)

    def test_gsd_narrow_fov(self):
        narrow_camera = _make_camera(fov_h=60.0)
        scaler = AltitudeScaler(100.0, 0.1, narrow_camera)
        # fov_h=60° → half-angle=30° → ground_width = 2*H*tan(30°)
        expected = 2 * 100.0 * math.tan(math.radians(30.0)) / 500
        gsd = scaler.compute_gsd(altitude_m=100.0, image_width_px=500)
        self.assertAlmostEqual(gsd, expected, places=6)

    def test_gsd_zero_altitude_raises(self):
        with self.assertRaises(ValueError):
            self.scaler.compute_gsd(altitude_m=0.0, image_width_px=1000)

    def test_gsd_negative_altitude_raises(self):
        with self.assertRaises(ValueError):
            self.scaler.compute_gsd(altitude_m=-50.0, image_width_px=1000)

    def test_gsd_zero_width_raises(self):
        with self.assertRaises(ValueError):
            self.scaler.compute_gsd(altitude_m=100.0, image_width_px=0)


class TestComputeScaleFactor(unittest.TestCase):

    def setUp(self):
        camera = _make_camera()
        self.scaler = AltitudeScaler(
            reference_altitude_m=100.0,
            reference_gsd_m=0.2,
            camera_config=camera,
        )

    def test_scale_at_reference_altitude_is_one(self):
        scale = self.scaler.compute_scale_factor(actual_altitude_m=100.0)
        self.assertAlmostEqual(scale, 1.0, places=6)

    def test_scale_doubles_at_double_altitude(self):
        # Higher drone → larger GSD → zoom in more (scale > 1)
        scale = self.scaler.compute_scale_factor(actual_altitude_m=200.0)
        self.assertAlmostEqual(scale, 2.0, places=6)

    def test_scale_halves_at_half_altitude(self):
        # Lower drone → smaller GSD → zoom out (scale < 1)
        scale = self.scaler.compute_scale_factor(actual_altitude_m=50.0)
        self.assertAlmostEqual(scale, 0.5, places=6)

    def test_scale_proportional_to_altitude(self):
        s1 = self.scaler.compute_scale_factor(actual_altitude_m=75.0)
        s2 = self.scaler.compute_scale_factor(actual_altitude_m=150.0)
        self.assertAlmostEqual(s2 / s1, 2.0, places=6)

    def test_scale_zero_altitude_raises(self):
        with self.assertRaises(ValueError):
            self.scaler.compute_scale_factor(actual_altitude_m=0.0)

    def test_scale_negative_altitude_raises(self):
        with self.assertRaises(ValueError):
            self.scaler.compute_scale_factor(actual_altitude_m=-10.0)


if __name__ == "__main__":
    unittest.main()
