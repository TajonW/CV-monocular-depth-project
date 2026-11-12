"""Publish selected stereo tables, figures, and portable run metadata."""

import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.stereo.core import read_pfm

plt.rcParams.update({"font.size": 13, "xtick.labelsize": 13, "ytick.labelsize": 13})


def load_scene(name):
    scene = ROOT / "data/middlebury/MiddEval3/trainingQ" / name
    with np.load(ROOT / "outputs/stereo/baseline" / name / "prediction.npz") as data:
        pred = {key: data[key] for key in data.files}
    gt = read_pfm(scene / "disp0GT.pfm")
    return scene, pred, gt


def main():
    tables = ROOT / "results/tables/stereo"
    figures = ROOT / "results/figures/stereo"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    runs = ("window3", "baseline", "window9")
    summaries = []
    for name in runs:
        run = ROOT / "outputs/stereo" / name
        shutil.copy2(run / "metrics.csv", tables / f"{name}_metrics.csv")
        summary = json.loads((run / "summary.json").read_text())
        config = json.loads((run / "manifest.json").read_text())["config"]
        summaries.append(dict(run=name, block_size=config["block_size"], **summary))
        shutil.copy2(run / "manifest.json", tables / f"{name}_manifest.json")
    (tables / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    fields = ["run", "block_size", "macro_mae_px", "macro_coverage", "macro_matcher_ms"]
    with (tables / "window_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    shutil.copy2(ROOT / "outputs/stereo/baseline/environment-freeze.txt", tables / "environment-freeze.txt")
    for name in ("Adirondack", "PianoL", "Jadeplant", "Pipes"):
        shutil.copy2(ROOT / "outputs/stereo/baseline" / name / "overview.png", figures / f"{name}_overview.png")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#dddddd")
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), layout="constrained")
    for i, name in enumerate(("Adirondack", "Teddy", "Jadeplant")):
        scene, data, gt = load_scene(name)
        good = np.isfinite(gt) & (gt > 0)
        vmax = np.percentile(gt[good], 99)
        finite = data["depth_m"][np.isfinite(data["depth_m"])]
        arrays = [cv2.imread(str(scene / "im0.png"))[..., ::-1],
                  np.where(good, gt, np.nan), data["disparity_px"], data["depth_m"]]
        for j, array in enumerate(arrays):
            ax = axes[i, j]
            upper = np.percentile(finite, 99) if j == 3 else vmax
            im = ax.imshow(array, cmap=cmap if j else None,
                           vmin=0 if j else None, vmax=upper if j else None)
            ax.set_title(f"{name}\n" + ("RGB", "GT disp. (px)", "Pred. disp. (px)", "Depth (m)")[j], fontsize=14)
            ax.axis("off")
            if j:
                fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    fig.savefig(figures / "qualitative.png", dpi=170)
    plt.close(fig)
    # Fixed, human-selected ROIs show actual lighting, thin-structure and boundary failures.
    rois = {"PianoL": (0.53, 0.02, 0.98, 0.60),
            "Jadeplant": (0.30, 0.04, 0.88, 0.61),
            "Pipes": (0.20, 0.28, 0.88, 0.85)}
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.8), layout="constrained")
    roi_meta = {}
    for i, (name, roi) in enumerate(rois.items()):
        scene, data, gt = load_scene(name)
        height, width = gt.shape
        x0, y0, x1, y1 = (int(roi[0]*width), int(roi[1]*height), int(roi[2]*width), int(roi[3]*height))
        crop = np.s_[y0:y1, x0:x1]
        left = cv2.imread(str(scene / "im0.png"))[..., ::-1]
        right = cv2.imread(str(scene / "im1.png"))[..., ::-1]
        valid = np.isfinite(gt) & (gt > 0) & data["valid_disparity"]
        error = np.where(valid, np.abs(data["disparity_px"] - gt), np.nan)
        arrays = [left, right, data["disparity_px"], error]
        upper = float(np.percentile(gt[np.isfinite(gt) & (gt > 0)], 99))
        for j, array in enumerate(arrays):
            ax = axes[i, j]
            cm = error_cmap if j == 3 else cmap if j == 2 else None
            im = ax.imshow(array[crop], cmap=cm, vmin=0 if j > 1 else None,
                           vmax=(10 if j == 3 else upper) if j > 1 else None)
            ax.set_title(f"{name}\n" + ("left RGB", "right RGB", "pred. disp. (px)", "error 0-10 px")[j], fontsize=14)
            ax.axis("off")
            if j > 1:
                fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
        roi_meta[name] = dict(x0=x0, y0=y0, x1=x1, y1=y1, coordinate_convention="same image-coordinate rectangle in each view; not correspondence aligned")
    fig.savefig(figures / "failure_cases.png", dpi=170)
    plt.close(fig)
    (figures / "failure_rois.json").write_text(json.dumps(roi_meta, indent=2), encoding="utf-8")
    print(f"Published tables: {tables}")
    print(f"Published figures: {figures}")


if __name__ == "__main__":
    main()
