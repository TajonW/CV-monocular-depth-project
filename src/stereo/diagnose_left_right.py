"""Add standalone LR diagnostics to an existing run without changing its files."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from .core import decode_disparity, evaluate_disparity, read_pfm
from .left_right import check_left_right
from .run_stereo import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["Left invalid", "Out of bounds", "Right invalid", "Inconsistent", "Consistent"]
COLORS = ["#d6d6d6", "#395c9a", "#e9ad35", "#d74b47", "#298c68"]


def draw_panel(left_rgb, left, right, result, name, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#dddddd")
    err_cm = plt.get_cmap("magma").copy()
    err_cm.set_bad("#dddddd")
    valid = np.isfinite(left)
    upper = max(1., float(np.nanpercentile(left, 99))) if valid.any() else 1
    axes[0, 0].imshow(left_rgb[..., ::-1])
    axes[0, 0].set_title("Left RGB")
    for ax, array, title in ((axes[0, 1], left, "Original left disparity (px)"),
                             (axes[0, 2], -right, "Right disparity magnitude (right view)"),
                             (axes[1, 2], np.where(result["consistent"], left, np.nan), "Left disparity passing LR (px)")):
        im = ax.imshow(array, cmap=cm, vmin=0, vmax=upper)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7)
    im = axes[1, 0].imshow(result["residual_px"], cmap=err_cm, vmin=0, vmax=5)
    axes[1, 0].set_title("LR residual in left view (display 0-5 px)")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.7)
    im = axes[1, 1].imshow(result["status"], cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(6)-0.5, 5))
    axes[1, 1].set_title("Five diagnostic states (left view)")
    colorbar = fig.colorbar(im, ax=axes[1, 1], ticks=range(5), shrink=0.8)
    colorbar.ax.set_yticklabels(LABELS, fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(name + " | Explicit left-right check, threshold 1 px by default", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/stereo/baseline")
    parser.add_argument("--output-dir", default="outputs/stereo/left_right")
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("threshold must be finite and nonnegative")
    run, output = (ROOT / args.run_dir).resolve(), (ROOT / args.output_dir).resolve()
    if output == run or run in output.parents:
        parser.error("diagnostic output must be separate from the baseline run")
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a fresh --output-dir")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if cv2.__version__ != manifest["packages"]["opencv-python"].rsplit(".", 1)[0]:
        parser.error("OpenCV version differs from the saved baseline")
    cv2.setNumThreads(manifest["opencv_threads"])
    dataset = ROOT / manifest["config"]["dataset_root"]
    # Verify the paired views are exactly those used in the baseline.
    for scene, hashes in manifest["input_sha256"].items():
        for name in ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm"):
            if sha256(dataset / scene / name) != hashes[name]:
                raise ValueError(f"Baseline input changed: {scene}/{name}")
    output.mkdir(parents=True, exist_ok=True)
    rows, source_hashes = [], {}
    for scene in sorted(manifest["input_sha256"]):
        original_path = run / scene / "prediction.npz"
        source_hashes[scene] = sha256(original_path)
        with np.load(original_path) as saved:
            left, left_valid = saved["disparity_px"], saved["valid_disparity"]
        params = json.loads((run / scene / "resolved_parameters.json").read_text())
        # Mirror the signed search interval [min, min+N-1] into [-min-N+1,-min].
        params["minDisparity"] = -params["minDisparity"] - params["numDisparities"] + 1
        matcher = cv2.StereoSGBM_create(**params)
        left_rgb = cv2.imread(str(dataset / scene / "im0.png"))
        right_rgb = cv2.imread(str(dataset / scene / "im1.png"))
        if left_rgb is None or right_rgb is None:
            raise ValueError("Unreadable stereo input")
        images = (right_rgb, left_rgb)
        if manifest["config"]["grayscale"]:
            images = tuple(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images)
        for _ in range(manifest["config"]["warmup_runs"]):
            matcher.compute(*images)
        durations = []
        for _ in range(manifest["config"]["timing_repeats"]):
            start = perf_counter()
            raw_right = matcher.compute(*images)
            durations.append((perf_counter()-start)*1000)
        right, right_valid = decode_disparity(raw_right, params["minDisparity"])
        start = perf_counter()
        result = check_left_right(left, left_valid, right, right_valid, args.threshold)
        check_ms = (perf_counter()-start)*1000
        gt = read_pfm(dataset / scene / "disp0GT.pfm")
        before = evaluate_disparity(left, left_valid, gt)
        after = evaluate_disparity(left, result["consistent"], gt)
        counts = np.bincount(result["status"].ravel(), minlength=5)
        count_checkable = int(result["checkable"].sum())
        row = dict(scene=scene, baseline_mae_px=before["mae_px"], baseline_coverage=before["coverage"],
                   lr_mae_px=after["mae_px"], lr_coverage=after["coverage"],
                   left_valid_pixels=int(left_valid.sum()), checkable_pixels=count_checkable,
                   consistent_pixels=int(counts[4]), inconsistent_pixels=int(counts[3]),
                   left_invalid_pixels=int(counts[0]), out_of_bounds_pixels=int(counts[1]),
                   right_invalid_pixels=int(counts[2]),
                   pass_fraction_checkable=float(counts[4]/count_checkable) if count_checkable else None,
                   mean_residual_px=float(np.mean(result["residual_px"][result["checkable"]])) if count_checkable else None,
                   right_matcher_ms=float(np.median(durations)), check_ms=check_ms)
        folder = output / scene
        folder.mkdir()
        np.savez_compressed(folder / "diagnostic.npz", raw_right_disparity=raw_right,
                            right_disparity_px=right, right_valid=right_valid, **result)
        cv2.imwrite(str(folder / "status.png"), result["status"])
        cv2.imwrite(str(folder / "inconsistent_mask.png"), (result["status"] == 3).astype(np.uint8)*255)
        cv2.imwrite(str(folder / "consistent_mask.png"), result["consistent"].astype(np.uint8)*255)
        write_json(folder / "right_parameters.json", params)
        write_json(folder / "metrics.json", row)
        draw_panel(left_rgb, left, right, result, scene, folder / "overview.png")
        rows.append(row)
        if sha256(original_path) != source_hashes[scene]:
            raise AssertionError("Baseline prediction was modified")
        print(f"{scene:14s} pass={row['pass_fraction_checkable']:.2%} MAE {before['mae_px']:.3f}->{after['mae_px']:.3f} coverage {before['coverage']:.2%}->{after['coverage']:.2%}", flush=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([r[key] for r in rows])) for key in
               ("baseline_mae_px", "baseline_coverage", "lr_mae_px", "lr_coverage", "pass_fraction_checkable", "right_matcher_ms", "check_ms")}
    summary.update(scene_count=len(rows), threshold_px=args.threshold,
                   sampling="nearest pixel: floor(xR+0.5); continuous xR must lie in [0,W-1]",
                   sign_convention="dL=xL-xR; dR=xR-xL; residual=abs(dL+dR(xL-dL,y)); inclusive threshold",
                   aggregation="equal per-scene means; MAE before/after uses different valid-pixel subsets",
                   interpretation="geometric consistency only, not ground-truth correctness or an occlusion label",
                   baseline_unchanged=True)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", dict(created_at=datetime.now().astimezone().isoformat(),
               command=sys.argv, source_run=str(run.relative_to(ROOT)), opencv=cv2.__version__, numpy=np.__version__,
               threshold_px=args.threshold, status_codes=dict(enumerate(LABELS)),
               source_prediction_sha256=source_hashes,
               source_manifest_sha256=sha256(run / "manifest.json"),
               code_sha256={f"src/stereo/{name}": sha256(ROOT / "src/stereo" / name) for name in ("left_right.py", "diagnose_left_right.py")}))
    tables = ROOT / "results/tables/stereo"
    figures = ROOT / "results/figures/stereo"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.csv", "summary.json", "manifest.json"):
        shutil.copy2(output / name, tables / ("left_right_" + name))
    for scene in ("Adirondack", "Jadeplant", "PianoL"):
        if (output / scene / "overview.png").exists():
            shutil.copy2(output / scene / "overview.png", figures / f"{scene}_left_right.png")
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.8), layout="constrained")
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("#dddddd")
    for i, scene in enumerate(("Adirondack", "Jadeplant", "PianoL")):
        rgb = cv2.imread(str(dataset / scene / "im0.png"))
        with np.load(output / scene / "diagnostic.npz") as data:
            status, residual = data["status"], data["residual_px"]
        axes[i, 0].imshow(rgb[..., ::-1])
        axes[i, 1].imshow(status, cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(6)-0.5, 5))
        im = axes[i, 2].imshow(residual, cmap=cm, vmin=0, vmax=5)
        fig.colorbar(im, ax=axes[i, 2], shrink=0.7)
        for j, label in enumerate(("RGB", "LR states", "Residual (px, 0-5)")):
            axes[i, j].set_title(scene + "\n" + label, fontsize=13)
            axes[i, j].axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=label) for c, label in zip(COLORS, LABELS)],
               loc="outside lower center", ncol=5, fontsize=9)
    fig.savefig(figures / "left_right_diagnostic.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
