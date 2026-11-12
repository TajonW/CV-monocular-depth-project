"""Metric-depth figures with a shared meter scale and explicit error scale."""

import cv2
from matplotlib import colormaps
import numpy as np


def save_figure(path, rgb, truth, prediction, row):
    images = [rgb]
    for depth in (truth, prediction):
        color = (colormaps["magma_r"](np.clip(depth / 10., 0, 1))[..., :3] * 255).astype(np.uint8)
        color[~np.isfinite(depth) | (depth <= 0)] = 0
        images.append(color)
    error = np.abs(np.clip(prediction, .001, 10.) - truth)
    error_rgb = (colormaps["inferno"](np.clip(error / 2., 0, 1))[..., :3] * 255).astype(np.uint8)
    error_rgb[~np.isfinite(truth) | (truth <= .001) | (truth >= 10)] = 0
    images.append(error_rgb)
    canvas = np.full((316, 1280, 3), 245, np.uint8)
    titles = ("RGB", "Filled GT | 0-10 m", "UniDepthV2-S | 0-10 m", "Absolute error | 0-2 m")
    for column, (image, title) in enumerate(zip(images, titles)):
        x = column * 320
        canvas[38:278, x:x + 320] = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, title, (x + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, .53, (35, 35, 35), 1, cv2.LINE_AA)
        if column:
            cv2.rectangle(canvas, (x + 20, 38 + 22), (x + 300, 38 + 235), (240, 240, 240), 1)
    label = (f"Test order {row['test_order']} | MATLAB index {row['matlab_index']} | "
             f"AbsRel {row['absrel']:.4f} | RMSE {row['rmse']:.4f} m | delta1 {row['delta1']:.4f} | "
             "white box: evaluation crop")
    cv2.putText(canvas, label, (8, 303), cv2.FONT_HERSHEY_SIMPLEX, .48, (35, 35, 35), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise OSError(path)
