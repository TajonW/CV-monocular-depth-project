"""Verify every saved prediction and the encoded RGB/depth video, without inference."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.monocular.run_framewise import sha256, write_json
from src.monocular.visualization import colorize_relative, comparison_image


def verify(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    normalization = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
    with (run / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    count, fps = summary["frames_processed"], summary["fps"]
    if len(rows) != count or manifest["frame_indices"] != list(range(count)):
        raise AssertionError("Frame count or ordering differs from manifest")
    if sorted(p.name for p in (run / "frames").iterdir()) != [f"{i:06d}" for i in range(count)]:
        raise AssertionError("Unexpected saved-frame directories")
    for path, expected in manifest["code_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise AssertionError(f"Inference source changed since the run: {path}")
    video_path = ROOT / manifest["input_video_path"]
    if sha256(video_path) != manifest["input_video_sha256"]:
        raise AssertionError("Input video changed")
    if sha256(ROOT / manifest["checkpoint_path"]) != manifest["checkpoint_sha256"]:
        raise AssertionError("Checkpoint changed")
    capture = cv2.VideoCapture(str(video_path))
    movie_paths = [run / summary["saved_video"]]
    compatible = run / "rgb_relative_depth_h264.mp4"
    if compatible.exists():
        movie_paths.append(compatible)
    movies = {p.name: cv2.VideoCapture(str(p)) for p in movie_paths}
    lower, upper = np.inf, -np.inf
    pixels, panels = 0, []
    frame_errors = {name: [] for name in movies}
    selected = {0, count//2, count-1}
    for name, movie in movies.items():
        if not movie.isOpened() or not np.isclose(movie.get(cv2.CAP_PROP_FPS), fps, atol=0.001):
            raise AssertionError(f"Output video cannot be opened or has wrong frame rate: {name}")
    try:
        for index, row in enumerate(rows):
            ok, source = capture.read()
            if not ok:
                raise AssertionError("Input ended before saved predictions")
            folder = run / "frames" / f"{index:06d}"
            saved_rgb = cv2.imread(str(folder / "rgb.png"))
            np.testing.assert_array_equal(source, saved_rgb)
            if int(row["frame_index"]) != index or not np.isclose(float(row["timestamp_s"]), index/fps):
                raise AssertionError("Frame index or timestamp mismatch")
            depth = np.load(folder / "relative_depth.npy", allow_pickle=False)
            if depth.dtype != np.float32 or depth.shape != source.shape[:2]:
                raise AssertionError("Raw depth dtype/shape mismatch")
            if not np.isfinite(depth).all() or (depth < 0).any():
                raise AssertionError("Nonfinite or negative prediction")
            lower, upper = min(lower, float(depth.min())), max(upper, float(depth.max()))
            pixels += depth.size
            color = colorize_relative(depth, normalization["minimum"], normalization["maximum"], normalization["palette"])
            np.testing.assert_array_equal(color, cv2.imread(str(folder / "relative_depth.png")))
            panel = comparison_image(source, color, index, index/fps)
            np.testing.assert_array_equal(panel, cv2.imread(str(folder / "comparison.png")))
            for name, movie in movies.items():
                ok, encoded = movie.read()
                if not ok or encoded.shape != panel.shape:
                    raise AssertionError(f"Encoded frame missing or wrong dimensions: {name}")
                # MP4 is lossy. PNGs are checked exactly; this catches severe video corruption.
                error = float(np.abs(encoded.astype(np.int16)-panel.astype(np.int16)).mean())
                if error > 20:
                    raise AssertionError(f"Severe encoded image mismatch at {name}, frame {index}: {error}")
                frame_errors[name].append(error)
            if index in selected:
                panels.append(cv2.resize(panel, (1280, round(panel.shape[0]*1280/panel.shape[1])), interpolation=cv2.INTER_AREA))
        input_extra_frame = capture.read()[0]
        for name, movie in movies.items():
            if movie.read()[0]:
                raise AssertionError(f"Output video contains extra frames: {name}")
    finally:
        capture.release()
        for movie in movies.values():
            movie.release()
    np.testing.assert_equal([lower, upper], [normalization["minimum"], normalization["maximum"]])
    if normalization["frame_count"] != count:
        raise AssertionError("Normalization does not cover the whole processed prefix")
    np.testing.assert_allclose(np.mean([float(r["inference_ms"]) for r in rows]), summary["mean_inference_ms"])
    figures, tables = ROOT / "results/figures/framewise", ROOT / "results/tables/framewise"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    contact = figures / f"{run.name}_contact_sheet.png"
    if not cv2.imwrite(str(contact), np.concatenate(panels, axis=0)):
        raise OSError("Cannot write contact sheet")
    result = dict(frames_verified=count, frame_indices=[0, count-1], fps=fps,
                  duration_s=count/fps, decoded_video_frames=count,
                  all_saved_rgb_match_source_in_order=True,
                  input_has_more_frames_after_prefix=bool(input_extra_frame),
                  raw_prediction_dtype="float32", raw_pixels_verified=pixels,
                  all_predictions_finite=True, exact_clip_wide_normalization=True,
                  exact_color_pngs_and_comparison_pngs=True,
                  normalization_minimum=lower, normalization_maximum=upper,
                  video_files_checked={name: dict(decoded_frames=len(errors), fps=fps,
                      mean_lossy_pixel_error=float(np.mean(errors)),
                      max_frame_lossy_pixel_error=float(max(errors)), sha256=sha256(run / name))
                      for name, errors in frame_errors.items()},
                  code_input_and_checkpoint_hashes_match=True,
                  course_100_frames_completed=False,
                  units="relative depth; not meters")
    write_json(run / "clip_verification.json", result)
    write_json(tables / f"{run.name}_clip_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/framewise/rollercoaster_70")
    verify(ROOT / parser.parse_args().run_dir)
