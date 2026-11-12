"""Evaluate raw RGB-only UniDepthV2 predictions under the course NYU protocol."""

import argparse
import csv
from datetime import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml

from src.monocular.run_framewise import sha256, write_json
from .data import NYU, load_splits
from .engine import MetricDepth
from .metrics import evaluate_image, CROP, MIN_DEPTH, MAX_DEPTH
from .visualization import save_figure

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metric.yaml")
    parser.add_argument("--run-name", default="nyu_test654")
    parser.add_argument("--max-images", type=int, default=654)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be one directory name")
    if not 1 <= args.max_images <= 654:
        parser.error("max-images must be within 1..654")
    config = yaml.safe_load((ROOT / args.config).read_text())
    dataset_path, splits, repository, checkpoint = (ROOT / config[key] for key in ("dataset", "splits", "repository", "checkpoint"))
    output = ROOT / config["output_dir"] / args.run_name
    if output.exists() and any(output.iterdir()):
        parser.error("Output nonempty; choose a new run-name")
    cv2.setNumThreads(config["opencv_threads"])
    train, test = load_splits(splits)
    indices = test[:args.max_images]
    dataset = NYU(dataset_path)
    first_rgb, _ = dataset.get(indices[0])
    print(f"Loading official UniDepthV2 ViT-S; requested {len(indices)}/654 test images", flush=True)
    started = perf_counter()
    engine = MetricDepth(repository, checkpoint)
    load_s = perf_counter() - started
    for _ in range(config["warmup_runs"]):
        engine.predict(first_rgb)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    (output / "predictions").mkdir()
    (output / "figures").mkdir()
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    model_source = read_json(repository / "source_provenance.json")
    weights = read_json(checkpoint / "download_provenance.json")
    code = sorted((ROOT / "src/metric").glob("*.py")) + [ROOT / "src/monocular/run_framewise.py", ROOT / "src/monocular/visualization.py"]
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), argv=sys.argv, config=config,
                    python=sys.version, executable=sys.executable, seed=0,
                    official_source=model_source, weight_download=weights,
                    dataset_sha256=sha256(dataset_path), splits_sha256=sha256(splits),
                    model_config_sha256=sha256(checkpoint / "config.json"), weights_sha256=sha256(checkpoint / "model.safetensors"),
                    official_code_sha256={str(p.relative_to(repository)): sha256(p) for p in sorted((repository / "unidepth").rglob("*.py"))},
                    project_code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code},
                    packages={p: metadata.version(p) for p in ("torch", "torchvision", "numpy", "opencv-python", "matplotlib", "PyYAML", "h5py", "scipy", "huggingface_hub", "safetensors", "timm", "pandas", "wandb", "termcolor", "tabulate", "trimesh")},
                    development_indices_zero_based=train.tolist(), full_test_indices_zero_based=test.tolist(),
                    evaluated_indices_zero_based=indices.tolist(),
                    data_access="images[i].transpose(2,1,0); filled depths[i].T; MATLAB split minus one",
                    input_policy="RGB uint8 only; official infer(rgb) default normalization/resize; no provided camera or GT inputs",
                    precision="Unmodified official infer CUDA FP16 autocast, FP32 weights; float32 raw depths saved in meters",
                    resolution_policy="Official default shape bounds, no resolution_level override", shape_constraints=engine.model.shape_constraints,
                    test_scale_alignment=False, tf32_enabled=False, strict_checkpoint_load=True,
                    protocol=dict(crop=list(CROP), min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, gt_range="strict inequalities",
                                  prediction_clipping=True, metrics=["absrel", "rmse", "delta1"], aggregation="equal per-image mean", delta1_threshold=1.25))
    write_json(output / "manifest.json", manifest)
    for relative in manifest["project_code_sha256"]:
        snapshot = output / "code_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, snapshot)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "environment-freeze.txt").write_text(freeze.stdout)
    rows, prediction_hashes = [], {}
    try:
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for order, index in enumerate(indices):
                rgb, truth = dataset.get(index)
                prediction, timing = engine.predict(rgb)
                metrics = evaluate_image(prediction, truth)
                path = output / "predictions" / f"{int(index):06d}.npy"
                np.save(path, prediction, allow_pickle=False)
                prediction_hashes[str(path.relative_to(output))] = sha256(path)
                row = dict(test_order=order, zero_based_index=int(index), matlab_index=int(index) + 1,
                           prediction_path=str(path.relative_to(output)), **metrics,
                           inference_pipeline_ms=timing["inference_pipeline_ms"],
                           network_height=timing["network_shape"][-2], network_width=timing["network_shape"][-1],
                           returned_dtype=timing["returned_dtype"], raw_min=float(prediction.min()), raw_max=float(prediction.max()),
                           finite_fraction=1.0, nonpositive_pixels=int(np.count_nonzero(prediction <= 0)))
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                if order == 0 or (order + 1) % 20 == 0 or order == len(indices) - 1:
                    print(f"{order + 1}/{len(indices)} | MATLAB index {int(index)+1} | AbsRel {metrics['absrel']:.5f} | "
                          f"pipeline {timing['inference_pipeline_ms']:.2f} ms", flush=True)
    except Exception as error:
        write_json(output / "failure.json", dict(completed_images=len(rows), error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        dataset.close()
    gpu = dict(name=torch.cuda.get_device_name(), total_mib=torch.cuda.get_device_properties(0).total_memory / 1024**2,
               peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
               peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2)
    summary = dict(status="completed", model="UniDepthV2 ViT-S", parameter_count=engine.parameter_count,
                   images_evaluated=len(rows), required_test_images=654, full_test_completed=len(rows) == 654,
                   metrics={key: float(np.mean([row[key] for row in rows])) for key in ("absrel", "rmse", "delta1")},
                   aggregation="Equal arithmetic mean of per-image metrics", protocol=manifest["protocol"],
                   mean_inference_pipeline_ms=float(np.mean([row["inference_pipeline_ms"] for row in rows])),
                   median_inference_pipeline_ms=float(np.median([row["inference_pipeline_ms"] for row in rows])),
                   timing_scope="Synchronized tensor preparation + official infer(rgb), normalization/resize, GPU compute, native-size postprocess, and depth float32 CPU copy; excludes HDF5 IO, metrics, writes, model load, and warm-up",
                   warmup_runs=config["warmup_runs"], model_load_s=load_s, gpu=gpu,
                   network_shapes=sorted({(row["network_height"], row["network_width"]) for row in rows}),
                   raw_output_size=[480, 640], raw_units="meters", precision=manifest["precision"],
                   all_predictions_finite=True, test_scale_alignment=False,
                   total_pred_clipped_low=sum(row["pred_clipped_low"] for row in rows),
                   total_pred_clipped_high=sum(row["pred_clipped_high"] for row in rows),
                   interpretation="Course-protocol local baseline, not a claim to reproduce a paper table exactly")
    write_json(output / "summary.json", summary)
    write_json(output / "prediction_sha256.json", prediction_hashes)
    fixed_orders = sorted({0, len(rows) // 2, len(rows) - 1})
    worst = sorted(rows, key=lambda row: row["absrel"], reverse=True)[:3]
    dataset = NYU(dataset_path)
    try:
        selected = [(rows[order], "representative") for order in fixed_orders] + [(row, "high_absrel") for row in worst]
        for row, group in selected:
            rgb, truth = dataset.get(row["zero_based_index"])
            prediction = np.load(output / row["prediction_path"], allow_pickle=False)
            save_figure(output / "figures" / f"{group}_{row['matlab_index']:04d}.png", rgb, truth, prediction, row)
    finally:
        dataset.close()
    write_json(output / "figure_selection.json", dict(fixed_test_orders=fixed_orders, high_absrel_cases=worst,
                                                     selection_note="Post-evaluation illustration only; no parameter selection/tuning"))
    figures, tables = ROOT / "results/figures/metric", ROOT / "results/tables/metric"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for path in (output / "figures").glob("*.png"):
        shutil.copy2(path, figures / f"{args.run_name}_{path.name}")
    for name in ("summary.json", "metrics.csv", "manifest.json", "figure_selection.json"):
        shutil.copy2(output / name, tables / f"{args.run_name}_{name}")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
