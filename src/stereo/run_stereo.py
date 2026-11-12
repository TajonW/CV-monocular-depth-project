"""Run the fixed CPU StereoSGBM baseline; execute from the repository root."""

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
import yaml

from .core import read_calibration, read_pfm, decode_disparity, disparity_to_depth, evaluate_disparity
from .visualization import scene_panel

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stereo.yaml")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.block_size is not None:
        config["block_size"] = args.block_size
    size = int(config["block_size"])
    if size < 1 or size % 2 == 0:
        parser.error("block size must be a positive odd integer")
    if config["warmup_runs"] < 0 or config["timing_repeats"] < 1:
        parser.error("invalid timing repeat counts")
    if config["method"] != "StereoSGBM" or config["mode"] != "SGBM":
        parser.error("this runner implements StereoSGBM in MODE_SGBM only")
    dataset = ROOT / config["dataset_root"]
    scenes = sorted(p for p in dataset.iterdir() if p.is_dir())
    if args.scenes:
        requested = set(args.scenes)
        if requested - {p.name for p in scenes}:
            parser.error("unknown scene names")
        scenes = [p for p in scenes if p.name in requested]
    elif len(scenes) != 15:
        raise ValueError(f"Expected 15 scenes, found {len(scenes)}")
    output = ROOT / config["output_dir"] / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Completed run exists: {output}; choose a new --run-name")
    cv2.setNumThreads(int(config["num_threads"]))
    rows, inputs = [], {}
    for scene in scenes:
        start = perf_counter()
        left = cv2.imread(str(scene / "im0.png"))
        right = cv2.imread(str(scene / "im1.png"))
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"Unreadable/mismatched image pair: {scene}")
        cal = read_calibration(scene / "calib.txt")
        if left.shape[:2] != (int(cal["height"]), int(cal["width"])):
            raise ValueError(f"Image/calibration size mismatch: {scene}")
        gt = read_pfm(scene / "disp0GT.pfm")
        if gt.shape != left.shape[:2]:
            raise ValueError("Ground truth dimensions differ from input")
        first, second = left, right
        if config["grayscale"]:
            first, second = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (left, right)]
        channels = 1 if first.ndim == 2 else first.shape[2]
        ndisp = config["num_disparities"]
        ndisp = int(np.ceil(cal["ndisp"] / 16) * 16) if ndisp == "auto_from_calibration" else int(ndisp)
        if ndisp <= 0 or ndisp % 16 or ndisp >= left.shape[1]:
            raise ValueError("Invalid disparity search range")
        parameters = dict(
            minDisparity=int(config["min_disparity"]), numDisparities=ndisp,
            blockSize=size, P1=int(config["p1_factor"] * channels * size**2),
            P2=int(config["p2_factor"] * channels * size**2),
            uniquenessRatio=int(config["uniqueness_ratio"]),
            speckleWindowSize=int(config["speckle_window_size"]),
            speckleRange=int(config["speckle_range"]),
            disp12MaxDiff=int(config["disp12_max_diff"]),
            preFilterCap=int(config["pre_filter_cap"]), mode=cv2.STEREO_SGBM_MODE_SGBM,
        )
        matcher = cv2.StereoSGBM_create(**parameters)
        for _ in range(config["warmup_runs"]):
            matcher.compute(first, second)
        durations = []
        for _ in range(config["timing_repeats"]):
            t0 = perf_counter()
            raw = matcher.compute(first, second)
            durations.append((perf_counter() - t0) * 1000)
        disp, valid = decode_disparity(raw, parameters["minDisparity"])
        depth, depth_valid = disparity_to_depth(disp, valid, cal)
        metrics = evaluate_disparity(disp, valid, gt)
        row = dict(scene=scene.name, width=left.shape[1], height=left.shape[0],
                   block_size=size, num_disparities=ndisp, **metrics,
                   matcher_ms=float(np.median(durations)),
                   depth_valid_pixels=int(depth_valid.sum()))
        folder = output / scene.name
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / "prediction.npz", raw_disparity=raw,
                            disparity_px=disp, depth_m=depth,
                            valid_disparity=valid, valid_depth=depth_valid)
        cv2.imwrite(str(folder / "invalid_mask.png"), (~valid).astype(np.uint8) * 255)
        row["processing_wall_s"] = perf_counter() - start
        write_json(folder / "metrics.json", row)
        write_json(folder / "resolved_parameters.json", parameters)
        scene_panel(left, right, gt, disp, depth, valid, row, folder / "overview.png", scene.name)
        rows.append(row)
        inputs[scene.name] = {name: sha256(scene / name) for name in ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm")}
        print(f"{scene.name:14s} MAE={row['mae_px']:.4f}px coverage={row['coverage']:.2%} match={row['matcher_ms']:.1f}ms", flush=True)
    keys = rows[0].keys()
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    total_eval = sum(r["evaluated_pixels"] for r in rows)
    summary = dict(
        scene_count=len(rows),
        macro_mae_px=float(np.mean([r["mae_px"] for r in rows])) if all(r["mae_px"] is not None for r in rows) else None,
        macro_coverage=float(np.mean([r["coverage"] for r in rows])),
        macro_matcher_ms=float(np.mean([r["matcher_ms"] for r in rows])),
        pooled_mae_px=sum(r["absolute_error_sum"] for r in rows) / total_eval if total_eval else None,
        pooled_coverage=total_eval / sum(r["gt_valid_pixels"] for r in rows),
        gpu_used=False, peak_gpu_memory_mib=0,
        timing="Median of 3 matcher.compute calls after 1 warm-up by default; excludes I/O, grayscale, evaluation, visualization; CPU, one OpenCV thread.",
    )
    summary["timing_repeats"] = config["timing_repeats"]
    summary["warmup_runs"] = config["warmup_runs"]
    write_json(output / "summary.json", summary)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        cpu_name = os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), command=sys.argv,
                    python=sys.version, executable=sys.executable, os=platform.platform(),
                    cpu=cpu_name, opencv_threads=cv2.getNumThreads(),
                    packages={p: metadata.version(p) for p in ("numpy", "opencv-python", "matplotlib", "PyYAML")},
                    git_commit=git.stdout.strip(), git_dirty=bool(status.stdout.strip()),
                    config=config, input_sha256=inputs,
                    code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src/stereo").glob("*.py"))})
    write_json(output / "manifest.json", manifest)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
