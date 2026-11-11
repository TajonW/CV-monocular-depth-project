"""Frame-independent Depth Anything V2-Small relative depth for a video prefix."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def inspect_video(path, count):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid source frame rate")
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        first = None
        for index in range(count):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Could not decode requested source frame {index}")
            if first is None:
                first = frame
            if frame.shape != first.shape:
                raise ValueError("Source frame dimensions change within the prefix")
        return first, dict(
            fps=fps, reported_frame_count=reported_count,
            source_size_hw=list(first.shape[:2]), verified_prefix_frames=count,
            sha256=sha256_file(path),
        )
    finally:
        capture.release()


@torch.inference_mode()
def infer_frame(model, frame, input_size, device):
    # Keep official OpenCV preprocessing and output interpolation.
    # image2tensor selects an upstream default device, then honor our explicit device.
    tensor, (height, width) = model.image2tensor(frame, input_size)
    tensor = tensor.to(device)
    prediction = model(tensor)
    prediction = functional.interpolate(
        prediction[:, None], (height, width), mode="bilinear", align_corners=True,
    )[0, 0]
    return prediction.float().cpu().numpy(), list(tensor.shape)


def render_video(video_path, output, rows, fps, limits, height, width):
    capture = cv2.VideoCapture(str(video_path))
    encoded_height = ((height + 48 + 1) // 2) * 2
    writer = cv2.VideoWriter(
        str(output / "rgb_depth.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width * 2, encoded_height),
    )
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise OSError("Cannot open source video or MP4 video writer")
        for row in rows:
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Decode failure while rendering frame {row['frame_index']}")
            depth = np.load(output / row["prediction_file"], allow_pickle=False)
            normalized = np.clip((depth - limits[0]) / (limits[1] - limits[0]), 0, 1)
            colors = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8),
                                      cv2.COLORMAP_TURBO)
            canvas = np.full((encoded_height, width * 2, 3), 255, np.uint8)
            canvas[48:48 + height, :width] = frame
            canvas[48:48 + height, width:] = colors
            cv2.putText(canvas, f"RGB | source frame {row['frame_index']:06d}",
                        (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Depth Anything V2-Small | relative depth",
                        (width + 12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            writer.write(canvas)
            if row["frame_index"] in {0, len(rows) // 2, len(rows) - 1}:
                if not cv2.imwrite(str(output / f"preview_{row['frame_index']:06d}.png"), canvas):
                    raise OSError("Could not save preview")
    finally:
        capture.release()
        writer.release()
    return [encoded_height, width * 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/videos/davis_rollercoaster.mp4"))
    parser.add_argument("--model-repo", type=Path, default=Path("external/Depth-Anything-V2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/depth_anything_v2_vits.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/framewise/dav2_small_100"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 100 or args.input_size <= 0 or args.cpu_threads <= 0:
        parser.error("Require 1..100 frames, positive input size and positive CPU thread count")
    first, video_info = inspect_video(args.video, args.frames)
    if args.inspect_only:
        print(json.dumps(video_info, indent=2))
        return
    if args.output_dir.exists():
        parser.error("Output directory already exists. Choose a new --output-dir for each run")
    if not args.checkpoint.is_file() or not (args.model_repo / "depth_anything_v2/dpt.py").is_file():
        parser.error("Prepare the official model repository and Small checkpoint first")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    sys.path.insert(0, str(args.model_repo.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    started = time.perf_counter()
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    load_seconds = time.perf_counter() - started
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    warmup_start = time.perf_counter()
    infer_frame(model, first, args.input_size, device)
    if device == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "predictions").mkdir()
    rows = []
    capture = cv2.VideoCapture(str(args.video))
    try:
        for index in range(args.frames):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Could not decode source frame {index}")
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            depth, tensor_shape = infer_frame(model, frame, args.input_size, device)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if depth.shape != frame.shape[:2] or depth.dtype != np.float32:
                raise ValueError(f"Unexpected prediction shape or dtype at frame {index}")
            if not np.isfinite(depth).all() or depth.min() < 0 or depth.max() <= depth.min():
                raise ValueError(f"Invalid or constant prediction at frame {index}")
            relative_file = f"predictions/depth_{index:06d}.npy"
            np.save(args.output_dir / relative_file, depth, allow_pickle=False)
            rows.append(dict(
                frame_index=index, timestamp_seconds=index / video_info["fps"],
                prediction_file=relative_file, height=depth.shape[0], width=depth.shape[1],
                dtype="float32", finite_fraction=1.0,
                minimum=float(depth.min()), maximum=float(depth.max()),
                input_tensor_shape=json.dumps(tensor_shape), inference_seconds=elapsed,
                source_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            ))
            print(f"frame {index:03d}: {elapsed:.3f} s, {depth.shape}, "
                  f"range [{depth.min():.4f}, {depth.max():.4f}]", flush=True)
    finally:
        capture.release()
    with (args.output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(args.output_dir / "frame_indices.npy", np.arange(args.frames, dtype=np.int64))
    limits = [min(row["minimum"] for row in rows), max(row["maximum"] for row in rows)]
    encoded_shape = render_video(args.video, args.output_dir, rows, video_info["fps"],
                                 limits, *video_info["source_size_hw"])
    metadata = dict(
        completed=True, model="Depth Anything V2-Small (vits), relative depth",
        model_repository="https://github.com/DepthAnything/Depth-Anything-V2",
        upstream_commit=git_value(args.model_repo, "rev-parse", "HEAD"),
        upstream_worktree_status=git_value(args.model_repo, "status", "--porcelain"),
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
        video=str(args.video), video_info=video_info,
        frame_count=args.frames, first_frame_index=0, last_frame_index=args.frames - 1,
        input_size_setting=args.input_size, input_tensor_shapes=sorted(set(row["input_tensor_shape"] for row in rows)),
        output_size_hw=video_info["source_size_hw"], encoded_video_size_hw=encoded_shape,
        device=device, precision="float32", cpu_threads=args.cpu_threads,
        hardware=platform.processor(), platform=platform.platform(),
        python=platform.python_version(), torch=torch.__version__, opencv=cv2.__version__,
        numpy=np.__version__, model_load_seconds=load_seconds, warmup_seconds=warmup_seconds,
        mean_inference_seconds=mean(row["inference_seconds"] for row in rows),
        total_inference_seconds=sum(row["inference_seconds"] for row in rows),
        estimated_100_frame_inference_seconds=100 * mean(row["inference_seconds"] for row in rows),
        timing_scope="official preprocessing, device transfer, forward, native-size interpolation and CPU copy; excludes decoding, saving and visualization; one separate warm-up",
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        visualization=dict(palette="OpenCV TURBO", minimum=limits[0], maximum=limits[1],
                           scope="one global min/max over all raw predictions in this run"),
        command=subprocess.list2cmdline([sys.executable, "-m", "src.monocular.run_framewise", *sys.argv[1:]]),
        run_wall_seconds=time.perf_counter() - started,
    )
    (args.output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Completed {len(rows)}/{args.frames}. Mean inference "
          f"{metadata['mean_inference_seconds']:.3f} s/frame; "
          f"100-frame estimate {metadata['estimated_100_frame_inference_seconds']:.1f} s.",
          flush=True)


if __name__ == "__main__":
    main()
