"""Explicit left-right consistency in left-view coordinates.

Both disparities are signed: dL = xL-xR, dR = xR-xL. Thus the residual is
abs(dL(xL,y) + dR(xL-dL,y)). No doffs is added to pixel correspondences.
"""

import numpy as np


def check_left_right(left, left_valid, right, right_valid, threshold=1.0):
    """Nearest-pixel sampling, fixed inclusive threshold, no extrapolation.

    status: 0 left invalid, 1 out of bounds, 2 right invalid,
    3 residual > threshold, 4 consistent. Errors are NaN if not checkable.
    """
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or not all(a.shape == left.shape for a in (left_valid, right, right_valid)):
        raise ValueError("All disparity/mask arrays must have the same 2D shape")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    height, width = left.shape
    x = np.broadcast_to(np.arange(width, dtype=np.float64), left.shape)
    y = np.broadcast_to(np.arange(height)[:, None], left.shape)
    valid_l = np.asarray(left_valid, dtype=bool) & np.isfinite(left)
    xr = x - np.where(valid_l, left, 0)
    inside = valid_l & (xr >= 0) & (xr <= width - 1)
    # Clipping is only for safe indexing; outside queries remain invalid.
    index = np.floor(np.clip(xr, 0, width - 1) + 0.5).astype(np.int64)
    sampled_right = right[y, index]
    checkable = inside & right_valid[y, index] & np.isfinite(sampled_right)
    residual = np.full(left.shape, np.nan, dtype=np.float32)
    residual[checkable] = np.abs(left[checkable] + sampled_right[checkable])
    consistent = checkable & (residual <= threshold)
    status = np.zeros(left.shape, dtype=np.uint8)
    status[valid_l & ~inside] = 1
    status[inside & ~checkable] = 2
    status[checkable & ~consistent] = 3
    status[consistent] = 4
    return dict(residual_px=residual, checkable=checkable,
                consistent=consistent, status=status)
