"""Independently re-evaluate saved floating-point predictions, without inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .core import read_pfm, evaluate_disparity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/stereo/baseline")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = root / args.run_dir
    config = yaml.safe_load((run / "config.yaml").read_text())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for scene in sorted(manifest["input_sha256"]):
        with np.load(run / scene / "prediction.npz") as data:
            metrics = evaluate_disparity(data["disparity_px"], data["valid_disparity"],
                read_pfm(root / config["dataset_root"] / scene / "disp0GT.pfm"))
        saved = json.loads((run / scene / "metrics.json").read_text())
        for key, value in metrics.items():
            if value is None:
                if saved[key] is not None:
                    raise AssertionError(f"{scene}: {key} differs")
            elif not np.isclose(value, saved[key], rtol=1e-10, atol=1e-10):
                raise AssertionError(f"{scene}: {key} differs")
        print(f"Verified {scene}: MAE={metrics['mae_px']:.4f}, coverage={metrics['coverage']:.4%}")
    print("All saved metrics reproduced from raw predictions.")


if __name__ == "__main__":
    main()
