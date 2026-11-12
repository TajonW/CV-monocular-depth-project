"""Independently recompute the course NYU metrics from every saved raw map."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monocular.run_framewise import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/metric/nyu_test654")
    args = parser.parse_args()
    output = ROOT / args.run_dir
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    hashes = json.loads((output / "prediction_sha256.json").read_text())
    config = manifest["config"]
    with (output / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary["images_evaluated"] == len(manifest["evaluated_indices_zero_based"])
    assert len(list((output / "predictions").glob("*.npy"))) == len(rows)
    assert len(hashes) == len(rows)
    for name, key in (("dataset", "dataset_sha256"), ("splits", "splits_sha256")):
        assert sha256(ROOT / config[name]) == manifest[key], name
    checkpoint = ROOT / config["checkpoint"]
    assert sha256(checkpoint / "model.safetensors") == manifest["weights_sha256"]
    assert sha256(checkpoint / "config.json") == manifest["model_config_sha256"]
    for relative, digest in manifest["official_code_sha256"].items():
        assert sha256(ROOT / config["repository"] / relative) == digest, relative
    for relative, digest in manifest["project_code_sha256"].items():
        assert sha256(output / "code_snapshot" / relative) == digest, relative
    split = loadmat(ROOT / config["splits"])
    test = split["testNdxs"].reshape(-1).astype(int) - 1
    train = split["trainNdxs"].reshape(-1).astype(int) - 1
    assert len(test) == 654 and len(train) == 795 and len(set(test) & set(train)) == 0
    assert list(test) == manifest["full_test_indices_zero_based"]
    assert list(test[:len(rows)]) == manifest["evaluated_indices_zero_based"]
    values = []
    valid_total, low_total, high_total = 0, 0, 0
    with h5py.File(ROOT / config["dataset"], "r") as dataset:
        for order, row in enumerate(rows):
            index = int(row["zero_based_index"])
            assert index == int(test[order]) and int(row["matlab_index"]) == index + 1 and int(row["test_order"]) == order
            path = output / row["prediction_path"]
            assert sha256(path) == hashes[row["prediction_path"]]
            prediction = np.load(path, allow_pickle=False)
            assert prediction.shape == (480, 640) and prediction.dtype == np.float32 and np.isfinite(prediction).all()
            truth = np.asarray(dataset["depths"][index]).T.astype(np.float64)
            mask = np.zeros((480, 640), dtype=bool)
            mask[45:471, 41:601] = True
            mask &= np.isfinite(truth) & (truth > .001) & (truth < 10.)
            gt = truth[mask]
            raw = prediction.astype(np.float64)[mask]
            pred = raw.clip(.001, 10.)
            absrel = float((np.abs(gt - pred) / gt).mean())
            rmse = float(np.linalg.norm(gt - pred) / np.sqrt(len(gt)))
            delta1 = float(np.count_nonzero((gt / pred < 1.25) & (pred / gt < 1.25)) / len(gt))
            measured = (absrel, rmse, delta1)
            np.testing.assert_allclose(measured, [float(row[k]) for k in ("absrel", "rmse", "delta1")], rtol=1e-10, atol=1e-12)
            assert int(mask.sum()) == int(row["valid_pixels"])
            assert np.count_nonzero(raw < .001) == int(row["pred_clipped_low"])
            assert np.count_nonzero(raw > 10.) == int(row["pred_clipped_high"])
            assert float(prediction.min()) == float(row["raw_min"]) and float(prediction.max()) == float(row["raw_max"])
            values.append(measured)
            valid_total += len(gt)
            low_total += int(row["pred_clipped_low"])
            high_total += int(row["pred_clipped_high"])
    average = dict(zip(("absrel", "rmse", "delta1"), np.mean(values, axis=0).tolist()))
    np.testing.assert_allclose(list(average.values()), [summary["metrics"][k] for k in average], rtol=1e-10, atol=1e-12)
    assert low_total == summary["total_pred_clipped_low"] and high_total == summary["total_pred_clipped_high"]
    assert summary["test_scale_alignment"] is False and manifest["test_scale_alignment"] is False
    result = dict(status="passed", images_verified=len(rows), full_654_test_verified=len(rows) == 654,
                  all_official_test_indices_in_order=True, raw_prediction_pixels_checked=len(rows) * 480 * 640,
                  valid_gt_pixels_checked=valid_total, hashes_and_source_snapshots_match=True,
                  all_predictions_float32_and_finite=True, independent_metrics_match_every_csv_row=True,
                  equal_per_image_average_matches_summary=True, no_test_scale_alignment=True,
                  recomputed_metrics=average)
    write_json(output / "verification.json", result)
    selected = ROOT / "results/tables/metric"
    selected.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "verification.json", selected / f"{output.name}_verification.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
