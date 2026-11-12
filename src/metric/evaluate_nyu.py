"""Evaluate UniDepthV2 ViT-S on the official NYU Depth V2 labeled split."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import h5py
import numpy as np
import psutil
from scipy.io import loadmat


MODEL_ID = "lpiccinelli/unidepth-v2-vits14"
MODEL_REVISION = "038c238f06c87b6c2f5b3749fd51fbf442b1f218"
WEIGHT_SHA256 = "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648"
UPSTREAM_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
CONFIG_SHA256 = "ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe"
IMAGE_SHAPE = (480, 640)
CROP = (slice(45, 471), slice(41, 601))
DEPTH_MIN, DEPTH_MAX = 0.001, 10.0
SPLIT_SIZES = {"dev": 795, "test": 654}
MIB = 1024**2


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def load_splits(path):
    names = {"dev": "trainNdxs", "test": "testNdxs"}
    data = loadmat(path, variable_names=tuple(names.values()))
    indices = {}
    for split, key in names.items():
        raw = data[key].ravel()
        if (raw.size != SPLIT_SIZES[split] or not np.issubdtype(raw.dtype, np.number)
                or not np.isfinite(raw).all() or np.any(raw != np.floor(raw))
                or np.any(raw < 1) or np.any(raw > sum(SPLIT_SIZES.values()))
                or np.unique(raw).size != raw.size):
            raise ValueError(f"{key} has invalid, duplicate or out-of-range MATLAB indices")
        indices[split] = raw.astype(np.int64) - 1
    if np.intersect1d(indices["dev"], indices["test"]).size:
        raise ValueError("NYU development and test indices overlap")
    return indices


def check_datasets(dataset):
    images, depths = dataset["images"], dataset["depths"]
    count = sum(SPLIT_SIZES.values())
    if images.shape != (count, 3, 640, 480) or images.dtype != np.uint8:
        raise ValueError(f"Unexpected NYU images shape or dtype: {images.shape}, {images.dtype}")
    if depths.shape != (count, 640, 480) or depths.dtype.kind != "f":
        raise ValueError(f"Unexpected NYU depths shape or dtype: {depths.shape}, {depths.dtype}")
    return images, depths


def read_rgb(images, index):
    return np.ascontiguousarray(images[index].transpose(2, 1, 0))


def evaluate_depth(prediction, truth):
    if prediction.shape != IMAGE_SHAPE or truth.shape != IMAGE_SHAPE:
        raise ValueError("Prediction and ground truth must be 480 x 640")
    if not np.isfinite(prediction).all():
        invalid = int(np.count_nonzero(~np.isfinite(prediction)))
        raise ValueError(f"Prediction contains {invalid} nonfinite values before clipping")
    cropped = truth[CROP]
    mask = np.isfinite(cropped) & (cropped >= DEPTH_MIN) & (cropped <= DEPTH_MAX)
    if not mask.any():
        raise ValueError("No valid ground-truth pixels in the course crop")
    gt = cropped[mask].astype(np.float64)
    pred = np.clip(prediction[CROP][mask].astype(np.float64), DEPTH_MIN, DEPTH_MAX)
    error = pred - gt
    return dict(
        valid_pixels=int(mask.sum()),
        absrel=float(np.mean(np.abs(error) / gt)),
        rmse_m=float(np.sqrt(np.mean(error**2))),
        delta1=float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25)),
    )


def save_preview(path, index, rgb, truth, prediction):
    import cv2

    def colors(values):
        normalized = np.clip(np.where(np.isfinite(values), values, 0), 0, 10) / 10
        return cv2.applyColorMap(np.rint(255 * normalized).astype(np.uint8), cv2.COLORMAP_TURBO)

    panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colors(truth), colors(prediction)]
    captioned = []
    for panel, label in zip(panels, (f"RGB | NYU index {index}", "Filled depth | 0-10 m", "UniDepthV2 | 0-10 m")):
        panel = cv2.copyMakeBorder(panel, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (41, 45 + 35), (600, 470 + 35), (255, 255, 255), 1)
        captioned.append(panel)
    if not cv2.imwrite(str(path), np.concatenate(captioned, axis=1)):
        raise OSError(f"Cannot write orientation preview: {path}")


def load_model(args):
    import torch

    git = ["git", "-C", str(args.model_repo)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    if commit != UPSTREAM_COMMIT or status:
        raise ValueError(f"Expected clean UniDepth checkout {UPSTREAM_COMMIT}, found {commit}")
    if sha256(args.checkpoint / "config.json") != CONFIG_SHA256:
        raise ValueError("UniDepth configuration does not match the official revision")
    checkpoint_hash = sha256(args.checkpoint / "model.safetensors")
    if checkpoint_hash != WEIGHT_SHA256:
        raise ValueError("UniDepth checkpoint SHA-256 does not match the official revision")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.model_repo.resolve()))
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained(str(args.checkpoint), local_files_only=True, strict=True)
    return model.to("cpu").eval(), checkpoint_hash


def predict(model, rgb):
    import torch
    from torch.nn import functional

    with torch.inference_mode():
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        depth = model.infer(tensor)["depth"]
        if depth.ndim != 4 or depth.shape[:2] != (1, 1):
            raise ValueError(f"Unexpected UniDepth depth shape: {tuple(depth.shape)}")
        if not torch.isfinite(depth).all().item():
            raise ValueError("UniDepth returned nonfinite depth before resizing")
        if tuple(depth.shape[-2:]) != IMAGE_SHAPE:
            depth = functional.interpolate(depth, size=IMAGE_SHAPE, mode="bilinear", align_corners=False)
        return depth[0, 0].float().numpy()


def memory_snapshot(process):
    info = process.memory_info()
    return dict(rss_mib=info.rss / MIB, peak_working_set_mib=getattr(info, "peak_wset", info.rss) / MIB,
                system_available_mib=psutil.virtual_memory().available / MIB)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args, images, depths, splits):
    indices = splits[args.split][:args.limit]
    process = psutil.Process()
    start = time.perf_counter()
    initial_memory = memory_snapshot(process)
    model, checkpoint_hash = load_model(args)
    model_load_seconds = time.perf_counter() - start
    warm_start = time.perf_counter()
    predict(model, read_rgb(images, int(splits["dev"][0])))
    warmup_seconds = time.perf_counter() - warm_start

    out = args.output
    (out / "predictions").mkdir(parents=True)
    rows = []
    memory_floor = min(initial_memory["system_available_mib"], memory_snapshot(process)["system_available_mib"])
    fields = ("split_position", "index_python", "index_matlab", "valid_pixels", "absrel", "rmse_m",
              "delta1", "prediction_min_m", "prediction_max_m", "prediction_file", "inference_seconds",
              "per_image_seconds", "rss_mib", "peak_working_set_mib", "system_available_mib")
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, index in enumerate(indices):
            index = int(index)
            image_start = time.perf_counter()
            rgb = read_rgb(images, index)
            truth = depths[index].T
            infer_start = time.perf_counter()
            prediction = predict(model, rgb)
            inference_seconds = time.perf_counter() - infer_start
            try:
                metrics = evaluate_depth(prediction, truth)
            except ValueError as error:
                raise ValueError(f"NYU Python index {index}: {error}") from error
            prediction_file = f"predictions/nyu_{index:04d}.npy"
            np.save(out / prediction_file, prediction, allow_pickle=False)
            memory = memory_snapshot(process)
            memory_floor = min(memory_floor, memory["system_available_mib"])
            row = dict(split_position=position, index_python=index, index_matlab=index + 1,
                       prediction_min_m=float(prediction.min()), prediction_max_m=float(prediction.max()),
                       prediction_file=prediction_file, inference_seconds=inference_seconds,
                       per_image_seconds=time.perf_counter() - image_start, **metrics, **memory)
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            if args.split == "dev" and position in (0, len(indices) // 2, len(indices) - 1):
                save_preview(out / f"preview_{index:04d}.jpg", index, rgb, truth, prediction)
            if position < 3 or (position + 1) % 10 == 0 or position + 1 == len(indices):
                print(f"{args.split} {position + 1}/{len(indices)} | NYU {index:04d} | "
                      f"{inference_seconds:.2f} s infer | peak {memory['peak_working_set_mib']:.0f} MiB | "
                      f"available {memory['system_available_mib']:.0f} MiB", flush=True)
            if args.split == "test" and memory["system_available_mib"] < 512:
                raise RuntimeError("System available memory fell below 512 MiB. Partial outputs were saved.")

    summary = dict(
        completed=True, split=args.split, count=len(rows),
        formal_test=(args.split == "test" and len(rows) == SPLIT_SIZES["test"]),
        mean_metrics={key: mean(row[key] for row in rows) for key in ("absrel", "rmse_m", "delta1")},
        aggregation="equal-weight per-image mean",
        model=MODEL_ID, model_revision=MODEL_REVISION, upstream_commit=UPSTREAM_COMMIT,
        weight_sha256=checkpoint_hash, config_sha256=sha256(args.checkpoint / "config.json"),
        dataset=str(args.dataset), splits=str(args.splits), splits_sha256=sha256(args.splits),
        image_size_hw=list(IMAGE_SHAPE), crop=[45, 471, 41, 601], depth_range_m=[DEPTH_MIN, DEPTH_MAX],
        mask="finite ground truth inside the crop and inclusive depth bounds",
        inference_input="RGB only, uint8 CHW, official infer preprocessing, default resolution bounds",
        raw_output="float32 480 x 640 before crop and value clipping",
        device="CPU", precision="float32", cpu_threads=args.cpu_threads,
        model_load_seconds=model_load_seconds, warmup_seconds=warmup_seconds,
        inference_timing_scope="RGB tensor conversion, model.infer, finite check, resize if required, float32 conversion",
        per_image_timing_scope="HDF5 read through prediction write and metrics, excluding previews",
        mean_inference_seconds=mean(row["inference_seconds"] for row in rows),
        median_inference_seconds=median(row["inference_seconds"] for row in rows),
        mean_per_image_seconds=mean(row["per_image_seconds"] for row in rows),
        estimate_654_seconds=654 * mean(row["per_image_seconds"] for row in rows),
        wall_seconds=time.perf_counter() - start,
        initial_available_mib=initial_memory["system_available_mib"],
        minimum_sampled_available_mib=memory_floor,
        peak_working_set_mib=max(row["peak_working_set_mib"] for row in rows),
        memory_method="Windows OS peak process working set (peak_wset); current RSS sampled per image",
        versions={name: importlib.metadata.version(name) for name in
                  ("torch", "torchvision", "numpy", "h5py", "scipy", "psutil", "huggingface-hub")},
        python=platform.python_version(), platform=platform.platform(),
        command=subprocess.list2cmdline([sys.executable, "-m", "src.metric.evaluate_nyu", *sys.argv[1:]]),
    )
    write_json(out / "run.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/nyu/nyu_depth_v2_labeled.mat"))
    parser.add_argument("--splits", type=Path, default=Path("data/nyu/splits.mat"))
    parser.add_argument("--model-repo", type=Path, default=Path("external/UniDepth"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/nyu/unidepth-v2-vits14"))
    parser.add_argument("--split", choices=tuple(SPLIT_SIZES), default="dev")
    parser.add_argument("--limit", type=int, help="Limit the development run to the first N official indices")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and (args.split != "dev" or not 1 <= args.limit <= SPLIT_SIZES["dev"]):
        parser.error("--limit is only for the development split and must be in 1..795")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not args.inspect_only and (args.output is None or args.output.exists()):
        parser.error("Specify a new --output directory")
    splits = load_splits(args.splits)
    print(f"Verified split sizes 795 development / 654 test, disjoint and zero-based. "
          f"First Python indices {int(splits['dev'][0])} / {int(splits['test'][0])}.", flush=True)
    with h5py.File(args.dataset, "r") as dataset:
        images, depths = check_datasets(dataset)
        if args.inspect_only:
            print(f"NYU images {images.shape} {images.dtype}, filled depths {depths.shape} {depths.dtype}")
        else:
            run(args, images, depths, splits)


if __name__ == "__main__":
    main()
