# Part 2: Metric depth

RGB-only UniDepthV2 ViT-S passed single-image inference and the complete
654-image NYU test evaluation. All saved predictions and metrics were checked
independently. These are local course-protocol results, not paper-table reproduction.

## Verified result

| Model | Images | AbsRel (lower) | RMSE m (lower) | delta1 (higher) |
|---|---:|---:|---:|---:|
| UniDepthV2 ViT-S, unchanged pretrained model | 654 | 0.114546 | 0.398888 | 0.892217 |

- RGB-only, no GT camera/depth or test-scale fitting used in inference.
- Official infer(rgb) preprocessing, default shape bounds, FP16 CUDA autocast,
  FP32 weights. Encoder input 1x3x490x644; saved depth 480x640 float32 meters.
- RTX 4070 Laptop: mean synchronized inference pipeline 25.8409 ms/image,
  median 25.3976 ms after two warm-ups. Includes tensor preparation, official
  normalization/resize, transfers, GPU compute, postprocess and depth CPU copy;
  excludes HDF5 IO, metrics, writes, initialization and warm-up.
- PyTorch peak allocated 484.958 MiB, reserved 602 MiB, including weights.
- All 200,908,800 raw values are finite; no images or hard pixels dropped.
  At valid cropped GT pixels, 540,915 predictions above 10 m were clipped for
  evaluation only. Original metric predictions are saved unchanged.

## Environment, data and weights

Use Python 3.11 and install the matching CUDA PyTorch build first (root README):

```powershell
python -m pip install -r requirements-metric.txt
python scripts/prepare_unidepth.py
```

Keep labeled data at `data/nyu/nyu_depth_v2_labeled.mat`. Preparation downloads
the official split if absent, pinned source to `external/UniDepth-source/`, and
Small weights to `checkpoints/unidepth-v2-vits14/`. Versions:

- [UniDepth source](https://github.com/lpiccinelli-eth/UniDepth), commit
  `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`.
- [ViT-S checkpoint](https://huggingface.co/lpiccinelli/unidepth-v2-vits14),
  revision `038c238f06c87b6c2f5b3749fd51fbf442b1f218`.
- [NYU split](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat),
  SHA256 `6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`.

The initial Git clone was slow; the fixed-commit source archive was used at
runtime. The clone eventually finished at the same commit but is unused by
this configuration. Both source versions are unmodified.

Linux-only training dependencies and compiled 3D KNN/patch extraction are not
needed for the course 2D metrics. Import may print warnings about KNN, training
loss acceleration or V1 NystromAttention. We use unchanged V2 inference and
its PyTorch SDPA fallback, with separate course metrics. W&B is disabled and
inference loads local weights only.

## Reproduce

From the project root, choose fresh names to preserve delivered runs:

```powershell
python -m src.metric.evaluate_nyu --run-name first_nyu_repro --max-images 1
python scripts/verify_nyu_run.py --run-dir outputs/metric/first_nyu_repro
python -m src.metric.evaluate_nyu --run-name nyu_test654_repro --max-images 654
python scripts/verify_nyu_run.py --run-dir outputs/metric/nyu_test654_repro
```

Delivered: `outputs/metric/first_nyu_image/` and `outputs/metric/nyu_test654/`.
`configs/metric.yaml` records paths, warm-ups and threads. No test-parameter search.

## Course protocol and verification

- Validate 795 development / 654 test indices, unique, disjoint and complete.
  Subtract one from MATLAB indices and preserve official test order.
- RGB: `images[i].transpose(2,1,0)`; filled GT: `depths[i].T`, not rawDepths.
- Evaluate native 480x640 output using crop `[45:471,41:601]`.
- GT mask: finite and strictly `.001 < depth < 10` meters. Prediction quality
  cannot change this mask. Nonfinite prediction causes explicit failure.
- Clip predictions to [.001,10] for metrics only; never fit scale to test GT.
- AbsRel, RMSE and delta1 (`max(pred/gt,gt/pred) < 1.25`) are computed per
  image, then averaged equally across all 654 images.

The independent verifier checks every prediction, GT, official index,
input/checkpoint/source/snapshot hash, recomputed CSV row and equal-image mean.
Four protocol tests cover scale fitting, crop/mask/clipping, strict delta1 and
nonfinite prediction rejection. First-image and full-run predictions match exactly.

## Artifacts and observed limitations

Each run contains raw `predictions/<zero-based-index>.npy`, `metrics.csv`,
summary, manifest, config, environment freeze, code snapshots, prediction hashes,
verification and figures. Selected outputs: `results/{figures,tables}/metric/`.
Panels show RGB / GT / prediction / absolute error. GT and prediction share
0-10 m coloring; error uses 0-2 m. White boxes indicate the metric crop.

Fixed representative examples are test orders 0/327/653. Three high-AbsRel
cases selected after evaluation, without parameter tuning:

- MATLAB index 469: doorway/interior; predicted distances are too large over
  broad surfaces and the distant opening loses GT depth variation.
  AbsRel 0.56291, RMSE 1.42891 m, delta1 0.08046.
- MATLAB index 88: shop/shelves; distant background overestimated, raw maximum
  14.326 m, with 51,224 valid cropped prediction pixels clipped above 10 m.
  AbsRel 0.39907, RMSE 1.72711 m, delta1 0.29181.
- MATLAB index 517: cluttered room; near-object boundaries and thin chair
  structures are smoothed. AbsRel 0.38099, RMSE 0.76389 m, delta1 0.58116.

These are observed differences from filled GT, not proven causes. Part 3 must
use development data for any calibration/tuning and preserve this baseline.
