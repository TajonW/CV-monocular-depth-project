"""Reproducible official offline VDA-Small inference with compatible video export."""

import argparse
from datetime import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import torch
import yaml

from src.monocular.run_framewise import probe_video, sha256, git_value, write_json
from .visualization import colorize_relative, comparison_image, save_video

ROOT = Path(__file__).resolve().parents[2]


def verify_video(path, count, fps, size):
    capture = cv2.VideoCapture(str(path))
    try:
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != size:
                raise RuntimeError(f"Wrong encoded size in {path}")
            decoded += 1
        if decoded != count or not np.isclose(actual_fps, fps, atol=0.001):
            raise RuntimeError(f"Wrong encoded count/FPS in {path}")
        return dict(frames=decoded, fps=actual_fps, size=list(size), sha256=sha256(path))
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/video.yaml")
    parser.add_argument("--run-name", default="smoke_32")
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be one directory name")
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    for option in ("input_size", "max_frames"):
        if getattr(args, option) is not None:
            config[option] = getattr(args, option)
    if (config["encoder"] != "vits" or config["precision"] != "fp16"
            or config["max_frames"] < 1 or config["input_size"] < 14 or config["input_size"] % 14):
        parser.error("Use Small, FP16, positive frame count, input-size a positive multiple of 14")
    if not torch.cuda.is_available():
        raise RuntimeError("This GPU experiment requires CUDA")
    video, checkpoint, repository = (ROOT / config[key] for key in ("video_path", "checkpoint", "repository"))
    output = ROOT / config["output_dir"] / args.run_name
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; use a fresh run-name")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _, video_info = probe_video(video, config["max_frames"])
    if not (repository / "video_depth_anything/video_depth.py").is_file():
        raise FileNotFoundError(repository)
    sys.path.insert(0, str(repository))
    from video_depth_anything import video_depth as official
    from utils import dc_utils
    if not all(Path(module.__file__).resolve().is_relative_to(repository.resolve())
               for module in (official, dc_utils)):
        raise ImportError("Unexpected official module import location")
    cv2.setNumThreads(config["opencv_threads"])
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    frames, fps = dc_utils.read_video_frames(str(video), config["max_frames"], -1, config["max_res"])
    if len(frames) != config["max_frames"] or not np.isclose(fps, video_info["fps"]):
        raise RuntimeError("Official reader changed the requested prefix or original FPS")
    # Independently confirm RGB order and the official reader's spatial resize.
    capture = cv2.VideoCapture(str(video))
    try:
        for index, rgb in enumerate(frames):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError("Video changed after preflight")
            expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if expected.shape != rgb.shape:
                expected = cv2.resize(expected, (rgb.shape[1], rgb.shape[0]))
            if not np.array_equal(expected, rgb):
                raise RuntimeError(f"Official reader RGB mismatch at frame {index}")
    finally:
        capture.release()
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_paths = sorted((repository / "video_depth_anything").rglob("*.py")) + sorted((repository / "utils").rglob("*.py"))
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), config=config,
                    argv=sys.argv, python=sys.version, executable=sys.executable,
                    official_git_commit=git_value(repository, "rev-parse", "HEAD"),
                    official_tracked_files_dirty=bool(git_value(repository, "status", "--porcelain", "--untracked-files=no")),
                    official_code_sha256={str(p.relative_to(repository)): sha256(p) for p in source_paths},
                    project_code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                         sorted((ROOT / "src/video").glob("*.py")) +
                                         [ROOT / "src/monocular/run_framewise.py", ROOT / "src/monocular/visualization.py"]},
                    packages={p: metadata.version(p) for p in ("torch", "torchvision", "numpy", "opencv-python", "matplotlib", "PyYAML", "einops", "easydict", "tqdm", "imageio", "imageio-ffmpeg")},
                    checkpoint_sha256=sha256(checkpoint), input_video_sha256=sha256(video),
                    source=video_info, decoder="decord" if dc_utils.DECORD_AVAILABLE else "OpenCV",
                    read_resolution=list(frames.shape[1:3]), fp16_policy="Official CUDA autocast; FP32 weights and returned arrays",
                    colormap_adaptation="Project save_video uses matplotlib.colormaps['inferno'].colors; unchanged official 255-index lookup",
                    ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version())
    write_json(output / "manifest.json", manifest)
    for relative in manifest["project_code_sha256"]:
        snapshot = output / "code_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, snapshot)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    model = official.VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to("cuda").eval()
    windows = []
    original_forward = model.forward

    def timed_forward(tensor):
        torch.cuda.synchronize()
        forward_start = perf_counter()
        windows.append(dict(input_shape=list(tensor.shape), input_dtype=str(tensor.dtype)))
        result = original_forward(tensor)
        torch.cuda.synchronize()
        windows[-1].update(forward_ms=(perf_counter() - forward_start) * 1000, output_dtype=str(result.dtype))
        return result

    # Official inference calls self.forward directly, bypassing nn.Module hooks.
    # Observe its arguments/results without changing tensor computations.
    model.forward = timed_forward
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    try:
        depths, returned_fps = model.infer_video_depth(frames, fps, input_size=config["input_size"], device="cuda", fp32=False)
        torch.cuda.synchronize()
    except Exception as error:
        write_json(output / "failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error),
                   config=config, windows=windows, peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2))
        raise
    finally:
        model.forward = original_forward
    pipeline_ms = (perf_counter() - start) * 1000
    gpu = dict(name=torch.cuda.get_device_name(), physical_total_mib=torch.cuda.get_device_properties(0).total_memory / 1024**2,
               peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
               peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2)
    expected_shape = (len(frames), *frames.shape[1:3])
    step = official.INFER_LEN - official.OVERLAP
    if len(windows) != (len(frames) + step - 1) // step:
        raise RuntimeError("Window instrumentation did not observe the official forwards")
    if (depths.shape != expected_shape or depths.dtype != np.float32 or not np.isfinite(depths).all()
            or np.any(depths < 0) or not np.isclose(returned_fps, fps)):
        raise RuntimeError("Invalid official relative-depth output; values were not silently masked")
    np.savez_compressed(output / "depths.npz", depths=depths)
    np.savez_compressed(output / "input_rgb.npz", frames=frames)
    lower, upper = float(depths.min()), float(depths.max())
    normalization = dict(palette="inferno", minimum=lower, maximum=upper, frame_count=len(frames),
                         scope="entire processed prefix", mode="global_minmax", units="relative; not meters",
                         lookup="official floor(normalized * 255) into 256-color table")
    write_json(output / "normalization.json", normalization)
    # Uses the new API adapter; external code/weights/temporal processing stay unchanged.
    save_video(depths, output / "relative_depth.mp4", fps, is_depths=True)
    contact = []
    with imageio.get_writer(str(output / "rgb_relative_depth.mp4"), fps=fps, macro_block_size=1,
                            codec="libx264", ffmpeg_params=["-crf", "18"]) as writer:
        for index, (rgb, depth) in enumerate(zip(frames, depths)):
            panel = comparison_image(rgb, colorize_relative(depth, lower, upper), index, fps)
            writer.append_data(panel)
            if index in {0, len(frames) // 2, len(frames) - 1}:
                contact.append(panel)
    contact_path = output / "contact_sheet.png"
    if not cv2.imwrite(str(contact_path), cv2.cvtColor(np.concatenate(contact), cv2.COLOR_RGB2BGR)):
        raise OSError(contact_path)
    checks = {name: verify_video(output / name, len(frames), fps, shape) for name, shape in (
        ("relative_depth.mp4", frames.shape[1:3]),
        ("rgb_relative_depth.mp4", (frames.shape[1] + 48, frames.shape[2] * 2 + 12)))}
    with np.load(output / "depths.npz") as saved:
        if not np.array_equal(depths, saved["depths"]):
            raise RuntimeError("Saved predictions differ from model output")
    step = official.INFER_LEN - official.OVERLAP
    summary = dict(status="passed", model="Video Depth Anything-Small", mode="offline relative depth", precision="fp16 autocast",
                   parameter_count=sum(p.numel() for p in model.parameters()), frames_processed=len(frames), frame_indices=[0, len(frames) - 1],
                   fps=fps, duration_s=len(frames) / fps, raw_shape=list(depths.shape), raw_dtype=str(depths.dtype),
                   original_resolution=[video_info["height"], video_info["width"]], read_resolution=list(frames.shape[1:3]),
                   input_size=config["input_size"], max_res=config["max_res"], windows=windows,
                   official_internal_tail_padding=(step - len(frames) % step) % step + official.OVERLAP,
                   padding_note="Official window-context padding only; returned output contains exactly the real requested prefix",
                   pipeline_total_ms=pipeline_ms, amortized_pipeline_ms_per_real_frame=pipeline_ms / len(frames),
                   timing_scope="Synchronized full official infer_video_depth: preprocessing, transfers, forwards, output resize, window alignment. Excludes decode, model load, visualization and file writes.",
                   warmup_runs=0, benchmark_note="Cold offline clip run without warm-up; not a steady-state speed benchmark",
                   gpu=gpu, all_predictions_finite=True, all_source_rgb_match_in_order=True, raw_npz_exact_roundtrip=True,
                   normalization=normalization, videos=checks, course_100_frames_completed=len(frames) == 100,
                   scope="Offline inference on the requested real frames; no temporal-quality or metric-accuracy conclusion")
    write_json(output / "summary.json", summary)
    figures, tables = ROOT / "results/figures/video", ROOT / "results/tables/video"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    shutil.copy2(contact_path, figures / f"{args.run_name}_contact_sheet.png")
    for name in ("summary.json", "manifest.json", "normalization.json"):
        shutil.copy2(output / name, tables / f"{args.run_name}_{name}")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
