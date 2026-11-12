"""Build and verify synchronized RGB / DA V2-Small / VDA-Small comparison."""

import argparse
from datetime import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

from src.monocular.run_framewise import sha256, write_json
from .visualization import colorize_relative

ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT, GAP, HEADER, FOOTER = 640, 360, 12, 72, 56


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def display_depth(depth, normalization):
    if depth.shape != (HEIGHT, WIDTH):
        depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return colorize_relative(depth, normalization["minimum"], normalization["maximum"])


def comparison_frame(rgb, da_depth, vda_depth, norms, index, fps):
    canvas = np.full((HEIGHT + HEADER + FOOTER, WIDTH * 3 + GAP * 2, 3), 245, np.uint8)
    rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    images = (rgb, display_depth(da_depth, norms["da2"]), display_depth(vda_depth, norms["vda"]))
    titles = (f"RGB | frame {index:06d} | {index / fps:.3f} s", "Depth Anything V2-Small", "Video Depth Anything-Small")
    subtitles = ("Same source frames | original frame rate", "Frame-wise | FP32 | relative depth", "Offline video | FP16 autocast | relative depth")
    footers = (f"Display: {WIDTH}x{HEIGHT} per panel | {fps:g} FPS",
               f"Fixed inferno range: {norms['da2']['minimum']:.3f} - {norms['da2']['maximum']:.3f}",
               f"Fixed inferno range: {norms['vda']['minimum']:.3f} - {norms['vda']['maximum']:.3f}")
    for column, (frame, title, subtitle, footer) in enumerate(zip(images, titles, subtitles, footers)):
        x = column * (WIDTH + GAP)
        canvas[HEADER:HEADER + HEIGHT, x:x + WIDTH] = frame
        for text, y, scale in ((title, 29, .65), (subtitle, 53, .5), (footer, HEADER + HEIGHT + 22, .5)):
            cv2.putText(canvas, text, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Relative depth: each model uses its own fixed full-clip scale; colors do not represent a shared distance in meters.",
                (12, HEADER + HEIGHT + 46), cv2.FONT_HERSHEY_SIMPLEX, .5, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def source_frames(da_run, video_run, norms, count, fps, source):
    """Validate source RGB order and raw predictions while yielding RGB canvases."""
    with np.load(video_run / "depths.npz", allow_pickle=False) as archive:
        vda = archive["depths"]
    with np.load(video_run / "input_rgb.npz", allow_pickle=False) as archive:
        video_rgb = archive["frames"]
    if (vda.dtype != np.float32 or vda.shape != video_rgb.shape[:3] or len(vda) != count
            or not np.isfinite(vda).all() or np.any(vda < 0)):
        raise ValueError("Invalid VDA arrays")
    if (float(vda.min()), float(vda.max())) != (norms["vda"]["minimum"], norms["vda"]["maximum"]):
        raise ValueError("VDA normalization does not match its entire raw clip")
    capture = cv2.VideoCapture(str(source))
    da_lower, da_upper = float("inf"), float("-inf")
    try:
        if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), fps):
            raise ValueError("Source video/FPS mismatch")
        for index in range(count):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"Source ends before frame {index}")
            folder = da_run / "frames" / f"{index:06d}"
            da = np.load(folder / "relative_depth.npy", allow_pickle=False)
            info = read_json(folder / "metadata.json")
            if info["frame_index"] != index or not np.isclose(info["timestamp_s"], index / fps):
                raise ValueError("DA frame index/timestamp mismatch")
            np.testing.assert_array_equal(bgr, cv2.imread(str(folder / "rgb.png")))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            expected_video_rgb = cv2.resize(rgb, (video_rgb.shape[2], video_rgb.shape[1]), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal(expected_video_rgb, video_rgb[index])
            if da.dtype != np.float32 or da.shape != bgr.shape[:2] or not np.isfinite(da).all() or np.any(da < 0):
                raise ValueError(f"Invalid DA raw frame {index}")
            da_lower, da_upper = min(da_lower, float(da.min())), max(da_upper, float(da.max()))
            yield index, comparison_frame(rgb, da, vda[index], norms, index, fps)
        if capture.read()[0]:
            raise ValueError("Source has additional frames; this command expects the full available clip")
        if (da_lower, da_upper) != (norms["da2"]["minimum"], norms["da2"]["maximum"]):
            raise ValueError("DA normalization does not match its entire raw clip")
    finally:
        capture.release()


def verify(output):
    manifest = read_json(output / "manifest.json")
    for relative, digest in manifest["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Source input changed: {relative}")
    for relative, digest in manifest["code_sha256"].items():
        if sha256(output / "code_snapshot" / relative) != digest:
            raise ValueError(f"Source snapshot changed: {relative}")
    count, fps = manifest["frames"], manifest["fps"]
    video_path = output / "rgb_da2_vda_comparison.mp4"
    capture = cv2.VideoCapture(str(video_path))
    errors = []
    try:
        if not np.isclose(capture.get(cv2.CAP_PROP_FPS), fps, atol=.001):
            raise ValueError("Encoded FPS mismatch")
        for index, expected in source_frames(ROOT / manifest["da2_run"], ROOT / manifest["vda_run"],
                                             manifest["normalization"], count, fps, ROOT / manifest["source_video"]):
            np.testing.assert_array_equal(cv2.cvtColor(expected, cv2.COLOR_RGB2BGR),
                                          cv2.imread(str(output / "frames" / f"{index:06d}.png")))
            ok, actual_bgr = capture.read()
            if not ok or actual_bgr.shape != expected.shape:
                raise ValueError(f"Encoded frame {index} missing or wrong size")
            error = np.abs(cv2.cvtColor(actual_bgr, cv2.COLOR_BGR2RGB).astype(np.int16) - expected.astype(np.int16))
            panel_errors = [float(error[HEADER:HEADER + HEIGHT, c * (WIDTH + GAP):c * (WIDTH + GAP) + WIDTH].mean()) for c in range(3)]
            if max(panel_errors) >= 20:
                raise ValueError(f"Encoded frame content mismatch at {index}: {panel_errors}")
            errors.append(panel_errors)
        if len(errors) != count or capture.read()[0]:
            raise ValueError("Encoded frame count mismatch")
    finally:
        capture.release()
    result = dict(status="passed", frames_verified=count, fps=fps, duration_s=count / fps,
                  encoded_size=[WIDTH * 3 + GAP * 2, HEIGHT + HEADER + FOOTER],
                  same_source_rgb_and_frame_order=True, source_has_more_frames=False,
                  exact_lossless_comparison_pngs=True, normalization_matches_both_entire_raw_clips=True,
                  both_methods_use_identical_color_lookup=True, inputs_and_source_snapshot_hashes_match=True,
                  mean_lossy_pixel_error_by_panel=dict(zip(("rgb", "da2", "vda"), np.mean(errors, axis=0).tolist())),
                  video_sha256=sha256(video_path), interpretation="Synchronized qualitative comparison; no accuracy conclusion")
    write_json(output / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framewise-run", default="outputs/framewise/rollercoaster_70")
    parser.add_argument("--video-run", default="outputs/video/rollercoaster_70_518")
    parser.add_argument("--output-dir", default="outputs/comparison/rollercoaster_70")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    if args.verify_only:
        print(json.dumps(verify(output), indent=2))
        return
    if output.exists() and any(output.iterdir()):
        parser.error("Output is nonempty; use a new --output-dir")
    da_run, video_run = ROOT / args.framewise_run, ROOT / args.video_run
    da_manifest, vda_manifest = read_json(da_run / "manifest.json"), read_json(video_run / "manifest.json")
    da_summary, vda_summary = read_json(da_run / "summary.json"), read_json(video_run / "summary.json")
    source = ROOT / da_manifest["input_video_path"]
    if sha256(source) != da_manifest["input_video_sha256"] or da_manifest["input_video_sha256"] != vda_manifest["input_video_sha256"]:
        raise ValueError("Models were not run on the same source video")
    count, fps = da_summary["frames_processed"], da_summary["fps"]
    if (count != vda_summary["frames_processed"] or not np.isclose(fps, vda_summary["fps"])
            or da_manifest["frame_indices"] != list(range(count)) or vda_summary["frame_indices"] != [0, count - 1]):
        raise ValueError("Model frame count, order or FPS differs")
    norms = dict(da2=read_json(da_run / "normalization.json"), vda=read_json(video_run / "normalization.json"))
    if any(n["palette"] != "inferno" or n["frame_count"] != count for n in norms.values()):
        raise ValueError("Normalization palette or scope mismatch")
    inputs = [source, video_run / "depths.npz", video_run / "input_rgb.npz"]
    for run in (da_run, video_run):
        inputs.extend(run / name for name in ("manifest.json", "normalization.json", "summary.json"))
    inputs.extend(p for index in range(count) for p in
                  (da_run / "frames" / f"{index:06d}" / name for name in ("relative_depth.npy", "rgb.png", "metadata.json")))
    code = [Path(__file__).resolve(), ROOT / "src/video/visualization.py", ROOT / "src/monocular/run_framewise.py", ROOT / "src/monocular/visualization.py"]
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), argv=sys.argv,
                    source_video=str(source.relative_to(ROOT)), da2_run=str(da_run.relative_to(ROOT)), vda_run=str(video_run.relative_to(ROOT)),
                    frames=count, fps=fps, display_panel_size=[WIDTH, HEIGHT], normalization=norms,
                    model_settings=dict(da2=dict(precision=da_manifest["config"]["precision"], raw_size=da_summary["original_resolution"], network_sizes=da_summary["network_input_shapes"]),
                                        vda=dict(precision=vda_manifest["config"]["precision"], raw_size=vda_summary["read_resolution"], network_sizes=sorted({tuple(w["input_shape"][-2:]) for w in vda_summary["windows"]}))),
                    display_policy="Resize scalar maps with OpenCV INTER_LINEAR to 640x360, then use identical official inferno 255-index lookup with each unchanged full-native-clip range. No per-frame normalization.",
                    input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
                    code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code},
                    packages={p: metadata.version(p) for p in ("numpy", "opencv-python", "matplotlib", "imageio", "imageio-ffmpeg")},
                    ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(),
                    inference_performed=False, metric_scale_shared=False)
    write_json(output / "manifest.json", manifest)
    for relative in manifest["code_sha256"]:
        snapshot = output / "code_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, snapshot)
    contact = []
    with imageio.get_writer(str(output / "rgb_da2_vda_comparison.mp4"), fps=fps, macro_block_size=1,
                            codec="libx264", ffmpeg_params=["-crf", "18"]) as writer:
        for index, panel in source_frames(da_run, video_run, norms, count, fps, source):
            writer.append_data(panel)
            if not cv2.imwrite(str(output / "frames" / f"{index:06d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
                raise OSError("Could not save comparison frame")
            if index in {0, count // 2, count - 1}:
                contact.append(panel)
    if not cv2.imwrite(str(output / "contact_sheet.png"), cv2.cvtColor(np.concatenate(contact), cv2.COLOR_RGB2BGR)):
        raise OSError("Could not save contact sheet")
    result = verify(output)
    figures, tables = ROOT / "results/figures/comparison", ROOT / "results/tables/comparison"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "contact_sheet.png", figures / f"{output.name}_contact_sheet.png")
    for name in ("manifest.json", "verification.json"):
        shutil.copy2(output / name, tables / f"{output.name}_{name}")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
