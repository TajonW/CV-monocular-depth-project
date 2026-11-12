"""Clip-wide depth visualization; raw predictions never use display scaling."""

import cv2
from matplotlib import colormaps
import numpy as np


def normalization_bounds(depth_paths):
    lower, upper = float("inf"), float("-inf")
    count = 0
    for path in depth_paths:
        depth = np.load(path, allow_pickle=False)
        if depth.size == 0 or not np.isfinite(depth).all():
            raise ValueError(f"Empty or nonfinite prediction: {path}")
        lower, upper = min(lower, float(depth.min())), max(upper, float(depth.max()))
        count += 1
    if not count:
        raise ValueError("No depth predictions to normalize")
    return lower, upper


def colorize_relative(depth, lower, upper, palette="inferno"):
    if not np.isfinite([lower, upper]).all() or upper < lower or not np.isfinite(depth).all():
        raise ValueError("Invalid depth or normalization range")
    normalized = (np.clip((depth.astype(np.float64)-lower)/(upper-lower), 0, 1)
                  if upper > lower else np.zeros(depth.shape, dtype=np.float64))
    rgb = (colormaps[palette](normalized)[..., :3]*255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def comparison_image(bgr, color_depth, frame_index, timestamp_s):
    if bgr.shape != color_depth.shape:
        raise ValueError("RGB and color depth dimensions must match")
    height, width = bgr.shape[:2]
    header, gap = 48, 12
    canvas = np.full((height+header, 2*width+gap, 3), 245, dtype=np.uint8)
    canvas[header:, :width] = bgr
    canvas[header:, width+gap:] = color_depth
    cv2.putText(canvas, f"RGB | frame {frame_index:06d} | {timestamp_s:.3f} s",
                (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Depth Anything V2-Small | relative depth",
                (width+gap+12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas
