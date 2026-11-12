"""Protocol tests catching scale fitting, crop leakage and invalid-pixel removal."""

import unittest
import numpy as np
from src.metric.metrics import evaluate_image


class MetricTests(unittest.TestCase):
    def test_absolute_scale_is_not_aligned(self):
        result = evaluate_image(np.full((480, 640), 2.), np.ones((480, 640)))
        self.assertEqual(result["absrel"], 1.)
        self.assertEqual(result["rmse"], 1.)
        self.assertEqual(result["delta1"], 0.)
        self.assertEqual(result["valid_pixels"], 426 * 560)

    def test_crop_gt_mask_and_prediction_clipping(self):
        truth = np.zeros((480, 640))
        prediction = np.full_like(truth, 20.)
        truth[100, 100] = 5.
        truth[100, 101] = 10.  # Strictly outside the GT depth range.
        truth[0, 0] = 1.  # Outside the spatial crop.
        result = evaluate_image(prediction, truth)
        self.assertEqual(result["valid_pixels"], 1)
        self.assertEqual(result["absrel"], 1.)
        self.assertEqual(result["rmse"], 5.)
        self.assertEqual(result["pred_clipped_high"], 1)

    def test_delta1_threshold_is_strict(self):
        result = evaluate_image(np.full((480, 640), 1.25), np.ones((480, 640)))
        self.assertEqual(result["delta1"], 0.)

    def test_nonfinite_prediction_is_not_dropped(self):
        prediction = np.ones((480, 640))
        prediction[100, 100] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite prediction"):
            evaluate_image(prediction, np.ones((480, 640)))


if __name__ == "__main__":
    unittest.main()
