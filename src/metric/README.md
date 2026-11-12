# NYU Depth V2 metric baseline

Part 2 uses the official UniDepthV2 ViT-S model
[`lpiccinelli/unidepth-v2-vits14`](https://huggingface.co/lpiccinelli/unidepth-v2-vits14)
with RGB-only `model.infer(rgb)`. The evaluator uses the official model
preprocessing and default resolution bounds.

## Prepare the data and model

Place these files under `data/nyu/`, reusing local copies when available.

- [NYU labeled MAT](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)
  contains `images` and filled `depths`. The file used for this baseline has
  SHA-256 `2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa`.
  A [pinned mirror of the MAT](https://huggingface.co/andrew-healey/nyuv2/blob/742f4d8a35f024263c6a9f7f316954b06ec5a817/nyu_depth_v2_labeled.mat)
  provides this checksum.
- [Official split MAT](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat)
  supplies the 795 development and 654 test indices. The evaluator verifies
  both counts, uniqueness and non-overlap, and subtracts one from every index.

Create an independent Python 3.10 environment in Windows Anaconda Prompt.
These commands run from the repository root.

```bat
conda activate cvproject3
python -m venv .venv\nyu-metric
.venv\nyu-metric\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-nyu.txt
```

For a fresh copy, prepare the pinned upstream checkout and checkpoint. Reuse
existing copies at the same revision if they are already present.

```bat
git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git external/UniDepth
git -C external/UniDepth fetch --depth 1 origin 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
git -C external/UniDepth checkout --detach 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
python -c "from huggingface_hub import snapshot_download; snapshot_download('lpiccinelli/unidepth-v2-vits14', revision='038c238f06c87b6c2f5b3749fd51fbf442b1f218', allow_patterns=['config.json','model.safetensors'], local_dir='checkpoints/nyu/unidepth-v2-vits14')"
```

The evaluator loads the official local safetensors strictly and verifies its
SHA-256. Set `WANDB_MODE=disabled` for local evaluation. UniDepth imports wandb
for its visualization utilities. The NYU run uses neither training logging
nor the optional KNN/xFormers extensions.

## Run the development pilot and official test

Choose new output directories when repeating a run.

```bat
set WANDB_MODE=disabled
set HF_HUB_OFFLINE=1
python -B -m src.metric.evaluate_nyu --inspect-only
python -B -m src.metric.evaluate_nyu --split dev --limit 15 --cpu-threads 4 --output outputs/nyu/unidepth_v2_vits14_dev15
python -B -m src.metric.evaluate_nyu --split test --cpu-threads 4 --output outputs/nyu/unidepth_v2_vits14_test654
```

The first development image is used for one separately timed warm-up in
each run. Development metrics are a protocol check. The formal result is
the equal-weight per-image mean over the 654 official test images.

The evaluator reads one RGB/depth pair at a time with h5py. It transposes
stored RGB to HWC and filled depth to HW, and passes only RGB in CHW order.
It saves the 480 × 640 float32 prediction before spatial cropping or value
clipping. Metrics use crop `[45:471, 41:601]`, finite ground truth in
`[0.001, 10]` m and predictions clipped to that same range. The evaluator
computes AbsRel, RMSE in meters and strict δ1 per image with no scale fitting.

Each run writes `per_image.csv`, `predictions/` and a final `run.json` in its
local output directory. The development run includes RGB and depth previews.
Inference time includes tensor conversion, the official inference call,
prediction validation and any resize. Per-image time also includes reading
the pair, computing metrics and saving its prediction. Process memory uses
the Windows peak working set, and system available memory is sampled per image.
