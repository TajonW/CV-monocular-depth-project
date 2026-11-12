"""Display scaling and input-length guards for the frame-wise baseline."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.monocular.visualization import normalization_bounds, colorize_relative
from src.monocular.run_framewise import probe_video

ROOT = Path(__file__).resolve().parents[1]


class FramewiseTests(unittest.TestCase):
    def test_normalization_uses_all_processed_frames(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            paths = [Path(folder) / "one.npy", Path(folder) / "two.npy"]
            np.save(paths[0], np.array([[0., 1.]], dtype=np.float32))
            np.save(paths[1], np.array([[100., 101.]], dtype=np.float32))
            self.assertEqual(normalization_bounds(paths), (0., 101.))
            first = colorize_relative(np.load(paths[0]), 0., 101.)
            isolated = colorize_relative(np.load(paths[0]), 0., 1.)
            self.assertFalse(np.array_equal(first[0, 1], isolated[0, 1]))

    def test_constant_depth_and_fixed_value_color(self):
        constant = colorize_relative(np.ones((2, 2), np.float32), 1., 1.)
        self.assertTrue(np.all(constant == constant[0, 0]))
        self.assertEqual(constant.dtype, np.uint8)
        a = colorize_relative(np.array([[1., 2.]]), 0., 10.)
        b = colorize_relative(np.array([[2., 9.]]), 0., 10.)
        np.testing.assert_array_equal(a[0, 1], b[0, 0])

    def test_nonfinite_depth_is_not_silently_hidden(self):
        with self.assertRaises(ValueError):
            colorize_relative(np.array([[np.nan]]), 0., 1.)
        with self.assertRaises(ValueError):
            colorize_relative(np.ones((1, 1)), 2., 1.)

    @unittest.skipUnless((ROOT / "data/videos/davis_rollercoaster.mp4").exists(), "Local clip not available")
    def test_video_prefix_and_no_fabricated_frames(self):
        first, info = probe_video(ROOT / "data/videos/davis_rollercoaster.mp4", 1)
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape[:2], (info["height"], info["width"]))
        # Ask for more than the local file can contain; must fail, not repeat/pad.
        with self.assertRaisesRegex(ValueError, "frames could be decoded"):
            probe_video(ROOT / "data/videos/davis_rollercoaster.mp4", info["reported_frame_count"] + 1)


if __name__ == "__main__":
    unittest.main()
