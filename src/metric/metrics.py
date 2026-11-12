"""Unaligned course metrics; prediction quality never changes the GT mask."""

import numpy as np

CROP = (45, 471, 41, 601)
MIN_DEPTH, MAX_DEPTH = 0.001, 10.0


def evaluate_image(prediction, ground_truth):
    if prediction.shape != (480, 640) or ground_truth.shape != (480, 640):
        raise ValueError("Evaluate native NYU height 480, width 640")
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite prediction; image/pixels were not silently excluded")
    y1, y2, x1, x2 = CROP
    truth = ground_truth[y1:y2, x1:x2].astype(np.float64)
    raw = prediction[y1:y2, x1:x2].astype(np.float64)
    mask = np.isfinite(truth) & (truth > MIN_DEPTH) & (truth < MAX_DEPTH)
    if not mask.any():
        raise ValueError("No valid ground truth in the course crop")
    truth, raw = truth[mask], raw[mask]
    pred = np.clip(raw, MIN_DEPTH, MAX_DEPTH)
    difference = pred - truth
    return dict(absrel=float(np.mean(np.abs(difference) / truth)),
                rmse=float(np.sqrt(np.mean(difference ** 2))),
                delta1=float(np.mean(np.maximum(pred / truth, truth / pred) < 1.25)),
                valid_pixels=int(len(truth)), pred_clipped_low=int(np.count_nonzero(raw < MIN_DEPTH)),
                pred_clipped_high=int(np.count_nonzero(raw > MAX_DEPTH)))
