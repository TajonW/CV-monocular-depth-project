"""Run independent Small-model inference on the first N video frames.

Default N=1 is a first-frame check, not the completed 100-frame course experiment.
All paths in the configuration are relative to the project root.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
import yaml

from .visualization import normalization_bounds, colorize_relative, comparison_image

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(directory, *arguments):
    result = subprocess.run(["git", "-C", str(directory), *arguments], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def probe_video(path, requested_frames):
    """Decode the requested prefix before model allocation; never duplicate frames."""
    if requested_frames < 1:
        raise ValueError("requested_frames must be positive")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open input video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Input video has no valid frame rate; no replacement FPS was assumed")
        info = dict(fps=fps, reported_frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                    decoder=capture.getBackendName(), requested_frames=requested_frames)
        first = None
        for index in range(requested_frames):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Only {index} frames could be decoded, but {requested_frames} were requested. "
                                 "Confirm the source clip; padding, looping, or resampling is not performed.")
            if first is None:
                first = frame
            elif frame.shape != first.shape:
                raise ValueError("Video frame dimensions changed")
        info.update(height=first.shape[0], width=first.shape[1])
        return first, info
    finally:
        capture.release()


def save_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main():
    # Keep video-length probing usable in CPU-only stereo test environments.
    import torch
    from .engine import FramewiseDepth

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/framewise.yaml")
    parser.add_argument("--run-name", default="first_frame")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    for option in ("max_frames", "input_size", "device", "precision"):
        if getattr(args, option) is not None:
            config[option] = getattr(args, option)
    if config["encoder"] != "vits" or config["max_frames"] < 1 or config["warmup_runs"] < 0:
        parser.error("Use the Small encoder, positive frame count and nonnegative warm-up count")
    output = ROOT / config["output_dir"] / args.run_name
    if output.exists() and any(output.iterdir()):
        parser.error("Run directory is not empty; choose a new --run-name")
    video, checkpoint, repository = (ROOT / config[key] for key in ("video_path", "checkpoint", "repository"))
    cv2.setNumThreads(config["opencv_threads"])
    torch.manual_seed(0)
    first_frame, video_info = probe_video(video, config["max_frames"])
    print(f"Video {video_info['width']}x{video_info['height']} at {video_info['fps']} FPS; "
          f"requested prefix: {config['max_frames']} frame(s)", flush=True)
    start = perf_counter()
    engine = FramewiseDepth(repository, checkpoint, config["input_size"], config["device"], config["precision"])
    engine.synchronize()
    model_load_s = perf_counter()-start
    for _ in range(config["warmup_runs"]):
        engine.predict(first_frame)
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    capture = cv2.VideoCapture(str(video))
    rows, depth_paths = [], []
    try:
        for index in range(config["max_frames"]):
            ok, frame = capture.read()
            if not ok or frame.shape != first_frame.shape:
                raise RuntimeError("Video changed after preflight validation")
            if index == 0 and not np.array_equal(frame, first_frame):
                raise RuntimeError("First-frame decoding was not repeatable")
            depth, timing = engine.predict(frame)
            folder = output / "frames" / f"{index:06d}"
            folder.mkdir(parents=True)
            save_png(folder / "rgb.png", frame)
            path = folder / "relative_depth.npy"
            np.save(path, depth, allow_pickle=False)
            depth_paths.append(path)
            row = dict(frame_index=index, timestamp_s=index/video_info["fps"],
                       height=frame.shape[0], width=frame.shape[1],
                       network_height=timing["input_tensor_shape"][2],
                       network_width=timing["input_tensor_shape"][3],
                       preprocess_ms=timing["preprocess_ms"], inference_ms=timing["inference_ms"],
                       to_cpu_ms=timing["to_cpu_ms"], pipeline_ms=timing["pipeline_ms"],
                       depth_min=float(depth.min()), depth_max=float(depth.max()),
                       depth_mean=float(depth.mean(dtype=np.float64)), finite_fraction=1.0)
            write_json(folder / "metadata.json", row)
            rows.append(row)
            print(f"Frame {index:06d}: {depth.shape}, {depth.dtype}, finite=100%, "
                  f"inference={timing['inference_ms']:.2f}ms", flush=True)
    finally:
        capture.release()
    engine.synchronize()
    gpu = dict(name=None, peak_allocated_mib=0.0, peak_reserved_mib=0.0)
    if engine.device.type == "cuda":
        gpu.update(name=torch.cuda.get_device_name(engine.device),
                   peak_allocated_mib=torch.cuda.max_memory_allocated(engine.device)/1024**2,
                   peak_reserved_mib=torch.cuda.max_memory_reserved(engine.device)/1024**2)
    # Second pass: establish one range across ALL processed frames before colorization.
    lower, upper = normalization_bounds(depth_paths)
    normalization = dict(mode="global_minmax_over_processed_frames", minimum=lower, maximum=upper,
                         palette=config["palette"], frame_count=len(rows), units="relative; not meters",
                         scope="first-frame validation only" if len(rows) == 1 else "entire processed prefix",
                         constant_map_policy="map to palette minimum when maximum equals minimum")
    write_json(output / "normalization.json", normalization)
    writer = None
    video_path = output / "rgb_relative_depth.mp4"
    try:
        for row, path in zip(rows, depth_paths):
            rgb = cv2.imread(str(path.parent / "rgb.png"))
            color = colorize_relative(np.load(path, allow_pickle=False), lower, upper, config["palette"])
            panel = comparison_image(rgb, color, row["frame_index"], row["timestamp_s"])
            save_png(path.parent / "relative_depth.png", color)
            save_png(path.parent / "comparison.png", panel)
            if row["frame_index"] == 0:
                save_png(output / "preview.png", panel)
            if config["save_video"] and len(rows) > 1:
                # Pad odd canvas dimensions for the encoder; native arrays are unchanged.
                panel = cv2.copyMakeBorder(panel, 0, panel.shape[0] % 2, 0, panel.shape[1] % 2,
                                           cv2.BORDER_CONSTANT, value=(245, 245, 245))
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             video_info["fps"], (panel.shape[1], panel.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Could not initialize MP4 writer")
                writer.write(panel)
    finally:
        if writer is not None:
            writer.release()
    if video_path.exists():
        validation = cv2.VideoCapture(str(video_path))
        try:
            encoded_fps = validation.get(cv2.CAP_PROP_FPS)
            count = 0
            while validation.read()[0]:
                count += 1
            if count != len(rows) or not np.isclose(encoded_fps, video_info["fps"], atol=0.001):
                raise RuntimeError("Encoded frame count or FPS differs from input prefix")
        finally:
            validation.release()
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    summary = dict(run_name=args.run_name, model="Depth Anything V2-Small", encoder="vits",
                   parameter_count=engine.parameter_count, device=str(engine.device), precision=engine.precision,
                   frames_processed=len(rows), first_frame_index=0, last_frame_index=len(rows)-1,
                   course_100_frames_completed=len(rows) == 100,
                   original_resolution=[video_info["height"], video_info["width"]],
                   network_input_shapes=sorted({(r["network_height"], r["network_width"]) for r in rows}),
                   fps=video_info["fps"], reported_video_frames=video_info["reported_frame_count"],
                   mean_inference_ms=float(np.mean([r["inference_ms"] for r in rows])),
                   mean_pipeline_ms=float(np.mean([r["pipeline_ms"] for r in rows])),
                   model_load_s=model_load_s, warmup_runs=config["warmup_runs"],
                   gpu=gpu, normalization=normalization, saved_video=video_path.name if video_path.exists() else None,
                   timing_scope="Synchronized forward plus output resize; pipeline additionally includes official transforms, host/device copies. Excludes decoding, model load, warm-up, visualization and disk writes.",
                   benchmark_note="One-frame validation latency, not a stable performance benchmark" if len(rows) == 1 else "One measured inference per processed frame after warm-up",
                   interpretation="Relative predictions only; no metric scale, accuracy metric or temporal-quality claim")
    write_json(output / "summary.json", summary)
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), argv=sys.argv, config=config,
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    packages={p: metadata.version(p) for p in ("torch", "torchvision", "numpy", "opencv-python", "matplotlib", "PyYAML")},
                    torch_cuda_runtime=torch.version.cuda, opencv_threads=cv2.getNumThreads(),
                    project_git_commit=git_value(ROOT, "rev-parse", "HEAD"),
                    project_git_dirty=bool(git_value(ROOT, "status", "--porcelain")),
                    official_repository=config["repository"],
                    official_git_commit=git_value(repository, "rev-parse", "HEAD"),
                    official_git_dirty=bool(git_value(repository, "status", "--porcelain", "--untracked-files=no")),
                    official_git_dirty_scope="tracked files only; generated Python bytecode is not a code change",
                    official_python_sha256={str(p.relative_to(repository)): sha256(p) for p in sorted((repository / "depth_anything_v2").rglob("*.py"))},
                    code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src/monocular").glob("*.py"))},
                    checkpoint_path=config["checkpoint"], checkpoint_sha256=sha256(checkpoint),
                    input_video_path=config["video_path"], input_video_sha256=sha256(video),
                    video=video_info, frame_indices=[r["frame_index"] for r in rows],
                    preprocessing="Official Resize lower_bound, keep_aspect_ratio, multiples of 14, OpenCV cubic, BGR->RGB/255, ImageNet normalize, CHW float32; native-size output with bilinear align_corners=True",
                    tf32_enabled=False, seed=0,
                    precision_policy="fp32 weights; fp16 uses CUDA autocast when explicitly requested",
                    xformers_available=bool(__import__("depth_anything_v2.dinov2_layers.attention", fromlist=["XFORMERS_AVAILABLE"]).XFORMERS_AVAILABLE))
    write_json(output / "manifest.json", manifest)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    if config["publish_preview"]:
        figures, tables = ROOT / "results/figures/framewise", ROOT / "results/tables/framewise"
        figures.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "preview.png", figures / f"{args.run_name}_preview.png")
        for name in ("summary.json", "manifest.json", "normalization.json"):
            shutil.copy2(output / name, tables / f"{args.run_name}_{name}")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
