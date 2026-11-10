"""CPU StereoSGBM baseline for Middlebury v3 quarter-resolution trainingQ."""

import argparse
import csv
import json
import platform
import re
import time
from statistics import mean
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_pfm(path):
    """Read a single-channel PFM, including endian flag and vertical flip."""
    with Path(path).open("rb") as f:
        def header_line():
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected end of PFM header: {path}")
                line = line.strip()
                if line and not line.startswith(b"#"):
                    return line
        if header_line() != b"Pf":
            raise ValueError(f"Expected a single-channel PFM: {path}")
        width, height = map(int, header_line().split())
        scale = float(header_line())
        if scale == 0:
            raise ValueError("PFM scale cannot be zero")
        values = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
    if values.size != width * height:
        raise ValueError(f"PFM size mismatch: {path}")
    return np.flipud(values.reshape(height, width)).astype(np.float32)


def read_calibration(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    matrix = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", values["cam0"])]
    if len(matrix) != 9:
        raise ValueError("cam0 must contain a 3x3 matrix")
    return {"focal_px": matrix[0], "baseline_mm": float(values["baseline"]),
            "doffs_px": float(values["doffs"]), "ndisp": int(values["ndisp"]),
            "width": int(values["width"]), "height": int(values["height"])}


def disparity_to_depth(disparity, valid, calib):
    denominator = disparity + calib["doffs_px"]
    usable = valid & np.isfinite(denominator) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[usable] = (calib["focal_px"] * calib["baseline_mm"] /
                     denominator[usable] / 1000.0)
    return depth


def evaluate(disparity, gt, prediction_valid):
    gt_valid = np.isfinite(gt) & (gt > 0)
    evaluation_valid = gt_valid & prediction_valid
    gt_count = int(gt_valid.sum())
    used_count = int(evaluation_valid.sum())
    if not gt_count or not used_count:
        raise ValueError("No valid pixels available for evaluation")
    mae = float(np.abs(disparity[evaluation_valid].astype(np.float64) -
                       gt[evaluation_valid].astype(np.float64)).mean())
    return {"mae_valid_px": mae, "valid_fraction_on_gt": used_count / gt_count,
            "gt_valid_pixels": gt_count, "evaluated_pixels": used_count}, evaluation_valid


def save_overview(left, disparity, gt, depth, valid, evaluation_valid, out, scene_name):
    # Use identical pixel scales for predicted and GT disparity. Invalid = black.
    gt_valid = np.isfinite(gt) & (gt > 0)
    vmax = max(float(np.percentile(gt[gt_valid], 99)), 1.0)
    error = np.full(gt.shape, np.nan, dtype=np.float32)
    error[evaluation_valid] = np.abs(disparity[evaluation_valid] - gt[evaluation_valid])
    error_max = max(float(np.percentile(error[evaluation_valid], 95)), 1.0)
    depth_values = depth[np.isfinite(depth)]
    if not depth_values.size:
        raise ValueError("No finite positive depth was recovered")
    depth_max = max(float(np.percentile(depth_values, 95)), 0.001)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    axes[0, 0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Left RGB")
    plots = [
        (axes[0, 1], np.where(valid, disparity, np.nan), "SGBM disparity (px)", vmax),
        (axes[0, 2], np.where(gt_valid, gt, np.nan), "GT disparity (px)", vmax),
        (axes[1, 1], error, "Absolute error on valid overlap (px)", error_max),
        (axes[1, 2], depth, "Depth from calibration (m)", depth_max),
    ]
    for ax, data, title, maximum in plots:
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=maximum)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[1, 0].imshow(~valid, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Invalid predictions (white = invalid)")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Middlebury {scene_name} | CPU StereoSGBM | Display ranges clip outliers")
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)


REQUIRED_INPUTS = ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm")
CSV_FIELDS = ("scene", "mae_px", "coverage_pct", "compute_ms")


def select_scenes(data_root, scene_name):
    if scene_name:
        scenes = [data_root / scene_name]
    else:
        scenes = sorted(path for path in data_root.iterdir() if path.is_dir())
        if len(scenes) != 15:
            raise ValueError(f"Expected 15 trainingQ scenes, found {len(scenes)}")
    missing = [scene / name for scene in scenes for name in REQUIRED_INPUTS
               if not (scene / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))
    return scenes


def matcher_parameters(config, calib, channels):
    ndisp = max(16, ((calib["ndisp"] + 15) // 16) * 16)
    if ndisp >= calib["width"]:
        raise ValueError("Disparity search range is too large for the image width")
    block_size = config["block_size"]
    penalty_scale = channels * block_size**2
    return {
        "minDisparity": config["min_disparity"],
        "numDisparities": ndisp,
        "blockSize": block_size,
        "P1": config["p1_factor"] * penalty_scale,
        "P2": config["p2_factor"] * penalty_scale,
        "disp12MaxDiff": config["disp12_max_diff"],
        "preFilterCap": config["pre_filter_cap"],
        "uniquenessRatio": config["uniqueness_ratio"],
        "speckleWindowSize": config["speckle_window_size"],
        "speckleRange": config["speckle_range"],
        "mode": getattr(cv2, f"STEREO_SGBM_MODE_{config['mode']}"),
    }


def run_scene(scene, output_root, config, legacy_positive_only):
    left = cv2.imread(str(scene / "im0.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(scene / "im1.png"), cv2.IMREAD_COLOR)
    if left is None or right is None or left.shape != right.shape:
        raise ValueError(f"{scene.name}: could not read matching left/right images")
    calib = read_calibration(scene / "calib.txt")
    gt = read_pfm(scene / "disp0GT.pfm")
    if gt.shape != left.shape[:2] or gt.shape != (calib["height"], calib["width"]):
        raise ValueError(f"{scene.name}: image, GT, and calibration sizes disagree")

    if config["grayscale"]:
        match_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        match_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        channels = 1
    else:
        match_left, match_right = left, right
        channels = left.shape[2]
    parameters = matcher_parameters(config, calib, channels)
    matcher = cv2.StereoSGBM_create(**parameters)
    start = time.perf_counter()
    raw = matcher.compute(match_left, match_right)
    elapsed = time.perf_counter() - start

    disparity = raw.astype(np.float32) / 16.0
    valid = np.isfinite(disparity) & (raw > (parameters["minDisparity"] - 1) * 16)
    zero_on_gt = int(np.count_nonzero(valid & (disparity == 0) & np.isfinite(gt) & (gt > 0)))
    validity_rule = "finite disparity and raw > (minDisparity - 1) * 16"
    if legacy_positive_only:
        valid &= disparity > 0
        validity_rule += " and disparity > 0"
    depth = disparity_to_depth(disparity, valid, calib)
    metrics, evaluation_valid = evaluate(disparity, gt, valid)
    metrics.update(
        scene=scene.name, image_size_hw=list(gt.shape), calibration=calib,
        parameters=parameters, sgbm_mode=config["mode"], grayscale=config["grayscale"],
        prediction_validity_rule=validity_rule,
        gt_support="finite GT disparity > 0; no non-occlusion mask applied",
        non_sentinel_zero_predictions_on_valid_gt=zero_on_gt,
        stereo_compute_seconds=elapsed,
        runtime_scope="one CPU matcher.compute call per new matcher, no explicit warm-up; excludes construction, I/O, grayscale conversion, evaluation, and plots",
        device="CPU", gpu_memory="not applicable (CPU execution)",
        platform=platform.platform(), processor=platform.processor(),
        python_version=platform.python_version(), opencv_version=cv2.__version__,
        numpy_version=np.__version__, opencv_threads=cv2.getNumThreads(),
    )
    out = output_root / scene.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "disparity_raw_x16.npy", raw)
    np.save(out / "disparity_px.npy", np.where(valid, disparity, np.nan))
    np.save(out / "depth_m.npy", depth)
    if not cv2.imwrite(str(out / "invalid_mask.png"), (~valid).astype(np.uint8) * 255):
        raise OSError(f"Could not write invalid mask to {out}")
    save_overview(left, disparity, gt, depth, valid, evaluation_valid, out, scene.name)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    return dict(scene=scene.name, mae_px=metrics["mae_valid_px"],
                coverage_pct=100 * metrics["valid_fraction_on_gt"], compute_ms=1000 * elapsed)


def write_table(output_root, rows):
    average = {"scene": "equal_weight_scene_mean"}
    for key in CSV_FIELDS[1:]:
        average[key] = mean(row[key] for row in rows)
    with (output_root / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in [*rows, average]:
            writer.writerow({**row, **{key: f"{row[key]:.6f}" for key in CSV_FIELDS[1:]}})
    print_row(average)


def print_row(row):
    print(f"{row['scene']}: MAE {row['mae_px']:.4f} px, coverage "
          f"{row['coverage_pct']:.2f}%, compute {row['compute_ms']:.2f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stereo.yaml"))
    parser.add_argument("--scene", help="Run one named trainingQ scene")
    parser.add_argument("--output-root", type=Path, help="Override the configured output directory")
    parser.add_argument("--legacy-positive-only", action="store_true",
                        help="Exclude zero disparity to reproduce the original single-scene rule")
    args = parser.parse_args()
    if args.scene and (Path(args.scene).name != args.scene or args.scene in (".", "..")):
        parser.error("--scene must be a scene name")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["method"] != "StereoSGBM" or config["num_disparities"] != "auto_from_calibration":
        parser.error("Expected StereoSGBM with num_disparities: auto_from_calibration")
    output_root = args.output_root or Path(config["output_dir"])
    if args.legacy_positive_only and args.output_root is None:
        output_root = output_root / "legacy_positive"
    scenes = select_scenes(Path(config["dataset_root"]), args.scene)
    rows = []
    for scene in scenes:
        row = run_scene(scene, output_root, config, args.legacy_positive_only)
        rows.append(row)
        print_row(row)
    if not args.scene:
        write_table(output_root, rows)
    print(f"Completed {len(rows)}/{len(scenes)} scenes in {output_root.resolve()}")


if __name__ == "__main__":
    main()
