"""Diagnostic plots; display clipping never changes quantitative predictions."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scene_panel(left, right, gt, pred, depth, valid, metrics, path, name):
    gt_valid = np.isfinite(gt) & (gt > 0)
    error = np.full(gt.shape, np.nan)
    mask = gt_valid & valid
    error[mask] = np.abs(pred[mask] - gt[mask])
    disparity_max = max(float(np.percentile(gt[gt_valid], 99)), 1)
    finite_depth = depth[np.isfinite(depth)]
    depth_max = float(np.percentile(finite_depth, 99)) if finite_depth.size else 1
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#eeeeee")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), layout="constrained")
    items = [
        (left[..., ::-1], "Left RGB", None, None),
        (right[..., ::-1], "Right RGB", None, None),
        (np.where(gt_valid, gt, np.nan), "GT disparity (px)", cm, disparity_max),
        (pred, "Predicted disparity (px)", cm, disparity_max),
        (depth, "Depth (m), display capped at P99", cm, depth_max),
        ((~valid).astype(float), "Invalid prediction (white)", "gray", 1),
        (error, "Absolute error (px), display 0-10", "magma", 10),
        (gt_valid.astype(float), "Valid GT (white)", "gray", 1),
    ]
    for ax, (array, title, cmap, upper) in zip(axes.flat, items):
        image = ax.imshow(array, cmap=cmap, vmin=0 if cmap else None, vmax=upper)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        if cmap and cmap != "gray":
            fig.colorbar(image, ax=ax, shrink=0.65)
    mae = metrics["mae_px"]
    fig.suptitle(f"{name} | MAE {mae:.3f} px | coverage {metrics['coverage']:.1%}", fontsize=15)
    fig.savefig(path, dpi=140)
    plt.close(fig)
