"""Verify saved VDA raw arrays, source order, color compatibility and encoded clips."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
from matplotlib import colormaps
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monocular.run_framewise import sha256, write_json
from src.video.visualization import colorize_relative, comparison_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    output = ROOT / args.run_dir
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    for key, config_key in (("checkpoint_sha256", "checkpoint"), ("input_video_sha256", "video_path")):
        assert sha256(ROOT / config[config_key]) == manifest[key], config_key
    for relative, digest in manifest["official_code_sha256"].items():
        assert sha256(ROOT / config["repository"] / relative) == digest, relative
    for relative, digest in manifest["project_code_sha256"].items():
        snapshot = output / "code_snapshot" / relative
        assert sha256(snapshot) == digest, relative
    with np.load(output / "input_rgb.npz") as saved:
        frames = saved["frames"]
    with np.load(output / "depths.npz") as saved:
        depths = saved["depths"]
    count, height, width = depths.shape
    assert count == config["max_frames"] == summary["frames_processed"]
    assert list(depths.shape) == summary["raw_shape"]
    assert depths.dtype == np.float32 and np.isfinite(depths).all() and np.all(depths >= 0)
    assert frames.dtype == np.uint8 and frames.shape == (count, height, width, 3)
    capture = cv2.VideoCapture(str(ROOT / config["video_path"]))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        for rgb in frames:
            ok, bgr = capture.read()
            assert ok
            expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if expected.shape != rgb.shape:
                expected = cv2.resize(expected, (width, height))
            np.testing.assert_array_equal(expected, rgb)
        source_has_more_frames = bool(capture.read()[0])
    finally:
        capture.release()
    normalization = json.loads((output / "normalization.json").read_text())
    lower, upper = float(depths.min()), float(depths.max())
    assert lower == normalization["minimum"] and upper == normalization["maximum"]
    assert np.isclose(fps, summary["fps"])
    for depth in depths:
        # Original official indexed-color formula; only registry lookup uses the new API.
        indices = ((depth - lower) / (upper - lower) * 255).astype(np.uint8) if upper > lower else np.zeros(depth.shape, np.uint8)
        reference = (np.asarray(colormaps["inferno"].colors)[indices] * 255).astype(np.uint8)
        np.testing.assert_array_equal(colorize_relative(depth, lower, upper), reference)
    videos = {}
    for name in ("relative_depth.mp4", "rgb_relative_depth.mp4"):
        path = output / name
        assert sha256(path) == summary["videos"][name]["sha256"]
        capture = cv2.VideoCapture(str(path))
        errors = []
        try:
            assert np.isclose(capture.get(cv2.CAP_PROP_FPS), fps, atol=0.001)
            for index, depth in enumerate(depths):
                ok, bgr = capture.read()
                assert ok
                expected = colorize_relative(depth, lower, upper)
                if name == "rgb_relative_depth.mp4":
                    expected = comparison_image(frames[index], expected, index, fps)
                actual = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                assert actual.shape == expected.shape
                error = float(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean())
                assert error < 20, (name, index, error)
                errors.append(error)
            assert not capture.read()[0], "Extra encoded frames"
        finally:
            capture.release()
        videos[name] = dict(frames=len(errors), fps=fps, mean_lossy_pixel_error=float(np.mean(errors)))
    step = 22
    if summary["windows"]:
        assert len(summary["windows"]) == (count + step - 1) // step
        assert all(window["input_shape"][:3] == [1, 32, 3] for window in summary["windows"])
    result = dict(status="passed", frames_verified=count, raw_pixels_verified=int(depths.size),
                  source_has_more_frames_after_prefix=source_has_more_frames,
                  all_source_rgb_match_in_order=True, raw_float32_finite_nonnegative=True,
                  clip_wide_normalization_exact=True, new_api_matches_official_color_lookup_exactly=True,
                  input_checkpoint_official_code_and_snapshot_hashes_match=True,
                  forward_windows_recorded=len(summary["windows"]), videos=videos)
    write_json(output / "verification.json", result)
    selected = ROOT / "results/tables/video"
    selected.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "verification.json", selected / f"{output.name}_verification.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
