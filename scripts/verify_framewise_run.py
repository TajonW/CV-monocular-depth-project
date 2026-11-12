"""Verify saved frame 0 against the official interface using the same checkpoint.

This runs additional inference on frame 0 only; it does not benchmark speed or
evaluate depth accuracy/temporal quality. Use the frame-wise project environment.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.monocular.engine import FramewiseDepth
from src.monocular.run_framewise import sha256, write_json
from src.monocular.visualization import colorize_relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/framewise/first_frame")
    args = parser.parse_args()
    run = ROOT / args.run_dir
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    for path, expected in manifest["code_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise AssertionError(f"Project inference code changed since this run: {path}")
    repository = ROOT / config["repository"]
    for path, expected in manifest["official_python_sha256"].items():
        if sha256(repository / path) != expected:
            raise AssertionError(f"Official model source changed: {path}")
    checkpoint = ROOT / config["checkpoint"]
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise AssertionError("Checkpoint changed")
    video = ROOT / config["video_path"]
    if sha256(video) != manifest["input_video_sha256"]:
        raise AssertionError("Input video changed")
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
        if not ok:
            raise AssertionError("Cannot decode source frame 0")
        actual_frames = 1
        while capture.read()[0]:
            actual_frames += 1
    finally:
        capture.release()
    folder = run / "frames/000000"
    np.testing.assert_array_equal(cv2.imread(str(folder / "rgb.png")), frame)
    saved = np.load(folder / "relative_depth.npy", allow_pickle=False)
    if saved.dtype != np.float32 or saved.shape != frame.shape[:2] or not np.isfinite(saved).all():
        raise AssertionError("Saved raw prediction has invalid dtype, shape or values")
    cv2.setNumThreads(config["opencv_threads"])
    engine = FramewiseDepth(repository, checkpoint, config["input_size"], summary["device"], config["precision"])
    own_tensor = engine.prepare(frame)
    reference_tensor, _ = engine.model.image2tensor(frame, config["input_size"])
    reference_tensor = reference_tensor.to(engine.device)
    np.testing.assert_array_equal(own_tensor.cpu().numpy(), reference_tensor.cpu().numpy())
    with torch.inference_mode(), engine.autocast_context():
        reference = engine.model.infer_image(frame, config["input_size"]).astype(np.float32)
    np.testing.assert_allclose(saved, reference, rtol=1e-5, atol=1e-6)
    normalization = json.loads((run / "normalization.json").read_text())
    color = colorize_relative(saved, normalization["minimum"], normalization["maximum"], normalization["palette"])
    np.testing.assert_array_equal(cv2.imread(str(folder / "relative_depth.png")), color)
    result = dict(frame_index=0, official_preprocessing_exact_match=True,
                  official_prediction_allclose=True,
                  max_absolute_prediction_difference=float(np.max(np.abs(saved-reference))),
                  saved_rgb_matches_decoded_frame=True, finite_fraction=float(np.isfinite(saved).mean()),
                  raw_dtype=str(saved.dtype), shape=list(saved.shape),
                  fixed_normalization_reproduced=True, input_checkpoint_and_code_hashes_match=True,
                  actual_decoded_video_frames=actual_frames,
                  requested_frames=config["max_frames"], full_100_frame_experiment_completed=False,
                  source_length_note="The supplied clip has fewer than 100 frames" if actual_frames < 100 else None)
    write_json(run / "verification.json", result)
    tables = ROOT / "results/tables/framewise"
    tables.mkdir(parents=True, exist_ok=True)
    write_json(tables / f"{run.name}_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
