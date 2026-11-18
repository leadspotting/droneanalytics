import unittest
from pathlib import Path

from PIL import Image

import api as api_module


class TestAltitudePreviewScaling(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("task5_test_tmp")
        self.tmp_dir.mkdir(exist_ok=True)
        self.paths = []
        for idx in range(2):
            path = self.tmp_dir / f"img_{idx}.png"
            Image.new("RGB", (100, 50), (20 * idx, 100, 150)).save(path)
            self.paths.append(path)

    def tearDown(self):
        for path in self.paths:
            if path.exists():
                path.unlink()
        if self.tmp_dir.exists():
            self.tmp_dir.rmdir()

    def _rows(self):
        base = {
            "lat": 32.08,
            "lon": 34.78,
            "heading": 0.0,
            "width_px": 100,
            "height_px": 50,
            "fov_h": 90.0,
            "fov_v": 60.0,
        }
        return [
            {**base, "path": str(self.paths[0]), "agl_m": 100.0},
            {**base, "path": str(self.paths[1]), "agl_m": 200.0, "lon": 34.781},
        ]

    def test_normalize_altitude_resizes_higher_altitude_image_larger(self):
        images, footprints = api_module._prepare_image_preview_inputs(
            self._rows(),
            scale=0.1,
            normalize_altitude=True,
        )
        self.assertEqual(len(images), 2)
        self.assertEqual(len(footprints), 2)
        self.assertEqual(images[0].shape[:2], (5, 10))
        self.assertEqual(images[1].shape[:2], (10, 20))

    def test_disable_normalize_altitude_keeps_same_preview_size(self):
        images, _ = api_module._prepare_image_preview_inputs(
            self._rows(),
            scale=0.1,
            normalize_altitude=False,
        )
        self.assertEqual(images[0].shape[:2], (5, 10))
        self.assertEqual(images[1].shape[:2], (5, 10))


if __name__ == "__main__":
    unittest.main()
