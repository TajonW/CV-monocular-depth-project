"""Calibration, PFM loading, depth conversion, and evaluation in pixel units."""

from pathlib import Path

import numpy as np


def read_calibration(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value.startswith("["):
            values[key.strip()] = np.array([
                [float(x) for x in row.split()]
                for row in value.strip("[]").split(";")
            ])
        else:
            values[key.strip()] = float(value)
    for key in ("cam0", "cam1", "baseline", "doffs", "width", "height", "ndisp"):
        if key not in values:
            raise ValueError(f"Missing {key} in {path}")
    if values["baseline"] <= 0 or values["cam0"][0, 0] <= 0:
        raise ValueError("Focal length and baseline must be positive")
    return values


def read_pfm(path):
    """Read Middlebury PFM pixel disparities; scale sign gives byte order.

    The header magnitude is display metadata, not a disparity unit conversion.
    E.g. Adirondack uses -0.003922 while payload values are already in pixels.
    """
    with Path(path).open("rb") as stream:
        def header_line():
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError(f"Truncated PFM header: {path}")
                line = line.strip()
                if line and not line.startswith(b"#"):
                    return line
        magic = header_line()
        if magic not in (b"Pf", b"PF"):
            raise ValueError(f"Invalid PFM signature: {path}")
        width, height = map(int, header_line().split())
        scale = float(header_line())
        if width <= 0 or height <= 0 or not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid PFM dimensions/scale: {path}")
        channels = 3 if magic == b"PF" else 1
        data = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
        if data.size != width * height * channels:
            raise ValueError(f"PFM payload size mismatch: {path}")
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(data.reshape(shape)).astype(np.float32).copy()


def decode_disparity(raw, min_disparity=0):
    disparity = raw.astype(np.float32) / 16.0
    valid = np.isfinite(disparity) & (raw > (min_disparity - 1) * 16)
    disparity[~valid] = np.nan
    return disparity, valid


def disparity_to_depth(disparity, valid, calibration):
    denominator = disparity.astype(np.float64) + calibration["doffs"]
    depth_valid = valid & np.isfinite(denominator) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[depth_valid] = (
        calibration["cam0"][0, 0] * calibration["baseline"]
        / denominator[depth_valid] / 1000.0
    )
    return depth, depth_valid


def evaluate_disparity(prediction, valid_prediction, ground_truth):
    if prediction.shape != ground_truth.shape or prediction.shape != valid_prediction.shape:
        raise ValueError("Prediction, validity mask, and ground truth shapes must match")
    gt_valid = np.isfinite(ground_truth) & (ground_truth > 0)
    valid = gt_valid & valid_prediction & np.isfinite(prediction)
    count_gt, count_eval = int(gt_valid.sum()), int(valid.sum())
    if count_gt == 0:
        raise ValueError("No finite, positive ground-truth disparities")
    errors = np.abs(prediction[valid].astype(np.float64) - ground_truth[valid])
    return {
        "gt_valid_pixels": count_gt,
        "evaluated_pixels": count_eval,
        "mae_px": float(errors.mean()) if count_eval else None,
        "coverage": count_eval / count_gt,
        "bad2_fraction": float((errors > 2.0).mean()) if count_eval else None,
        "absolute_error_sum": float(errors.sum()),
    }
