"""Critical unit/coordinate/masking checks for the scientific pipeline."""

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from src.stereo.core import read_pfm, decode_disparity, disparity_to_depth, evaluate_disparity
from src.stereo.left_right import check_left_right


class StereoTests(unittest.TestCase):
    def test_pfm_orientation_and_endianness(self):
        expected = np.array([[1, 2], [3, np.inf]], np.float32)
        temp_root = Path(__file__).resolve().parents[1] / "tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            for endian, scale in (("<f4", -1), (">f4", 1), ("<f4", -0.003922)):
                path = Path(folder) / "example.pfm"
                path.write_bytes(f"Pf\n2 2\n{scale}\n".encode() + np.flipud(expected).astype(endian).tobytes())
                np.testing.assert_array_equal(read_pfm(path), expected)

    def test_fixed_point_and_invalid_zero_disparity(self):
        d, valid = decode_disparity(np.array([[-16, 0, 160]], np.int16))
        np.testing.assert_array_equal(valid, [[False, True, True]])
        self.assertTrue(np.isnan(d[0, 0]))
        self.assertEqual(d[0, 2], 10)

    def test_depth_offset_and_millimeters(self):
        cal = {"cam0": np.diag([1000., 1000., 1.]), "baseline": 100., "doffs": 10.}
        d = np.array([[40., -10., np.nan]], np.float32)
        z, mask = disparity_to_depth(d, np.ones_like(d, bool), cal)
        self.assertEqual(z[0, 0], 2.)
        np.testing.assert_array_equal(mask, [[True, False, False]])
        self.assertTrue(np.isnan(z[0, 1]))

    def test_metric_denominators(self):
        gt = np.array([[10., 20., 30., 0., np.inf]])
        pred = np.array([[12., 18., np.nan, 100., 100.]])
        m = evaluate_disparity(pred, np.isfinite(pred), gt)
        self.assertEqual(m["mae_px"], 2.)
        self.assertAlmostEqual(m["coverage"], 2 / 3)

    def test_synthetic_rectified_translation(self):
        rng = np.random.default_rng(0)
        left = rng.integers(0, 256, (80, 192), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32,
                                      blockSize=3, P1=72, P2=288)
        d, mask = decode_disparity(matcher.compute(left, right))
        region = d[10:-10, 40:-20]
        self.assertAlmostEqual(float(np.nanmedian(region)), 8., delta=0.25)

    def test_left_right_categories_and_no_border_wrapping(self):
        left = np.full((1, 8), 2., dtype=np.float32)
        right = np.full_like(left, -2.)
        valid_left = np.ones_like(left, bool)
        valid_left[0, 0] = False
        valid_right = np.ones_like(left, bool)
        valid_right[0, 2] = False
        right[0, 3] = -5.
        result = check_left_right(left, valid_left, right, valid_right)
        np.testing.assert_array_equal(result["status"], [[0, 1, 4, 4, 2, 3, 4, 4]])
        self.assertTrue(np.isnan(result["residual_px"][0, 1]))
        self.assertEqual(result["residual_px"][0, 5], 3.)

    def test_left_right_zero_subpixel_threshold_and_nan(self):
        left = np.zeros((1, 8), dtype=np.float32)
        right = np.zeros_like(left)
        valid = np.ones_like(left, bool)
        left[0, 4] = 2.4  # xR=1.6 => nearest index 2.
        right[0, 2] = -3.4
        left[0, 6] = np.nan
        result = check_left_right(left, valid, right, valid, threshold=1.0)
        self.assertEqual(result["status"][0, 0], 4)  # Zero disparity is valid.
        self.assertAlmostEqual(result["residual_px"][0, 4], 1.)
        self.assertTrue(result["consistent"][0, 4])
        self.assertEqual(result["status"][0, 6], 0)
        with self.assertRaises(ValueError):
            check_left_right(left, valid, right, valid, threshold=-1)

    def test_synthetic_bidirectional_signed_matching(self):
        rng = np.random.default_rng(2)
        left = rng.integers(0, 256, (80, 192), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        params = dict(numDisparities=32, blockSize=3, P1=72, P2=288)
        dl, vl = decode_disparity(cv2.StereoSGBM_create(minDisparity=0, **params).compute(left, right), 0)
        dr, vr = decode_disparity(cv2.StereoSGBM_create(minDisparity=-31, **params).compute(right, left), -31)
        self.assertAlmostEqual(float(np.nanmedian(dr[10:-10, 40:-40])), -8., delta=0.25)
        result = check_left_right(dl, vl, dr, vr)
        self.assertGreater(float(result["consistent"][10:-10, 40:-40].mean()), 0.95)


if __name__ == "__main__":
    unittest.main()
