# Experiment log

## 2026-11-10 — Middlebury trainingQ StereoSGBM (CPU)

- Revision: `part1/stereo-baseline` working tree based on `cefa4bf`.
  The baseline migrates the validated local `stereo_single.py`.
- Environment: `cvproject3` on Windows build 22000, Intel Core i5-1135G7 CPU.
  Python 3.10.22, OpenCV 5.0.0 (8 reported threads), NumPy 2.2.6,
  Matplotlib 3.10.9, PyYAML 6.0.3. CPU only, GPU memory is inapplicable.
- Data: all 15 native quarter-resolution trainingQ pairs. Each scene's
  `im0.png`, `im1.png`, `calib.txt` and `disp0GT.pfm` matched the validated
  source by SHA-256 (60/60 files). The PFM reader preserves the original
  endian-sign handling and vertical flip. The scale field is `-0.003922`
  in these inputs and the payload is already in pixels.
  [Middlebury PFM format](https://vision.middlebury.edu/stereo/submit3/upload-format.html)
- SGBM: the previous YAML values match the legacy script. The YAML now
  explicitly includes the three legacy settings it previously omitted:
  `SGBM_3WAY`, `disp12MaxDiff=1`, and `preFilterCap=31`. Grayscale,
  minimum disparity 0, block size 5, P1=200, P2=800, uniqueness 10,
  speckle window 100 and range 2. Search width is calibrated `ndisp`
  rounded up to a multiple of 16, with a minimum of 16.
- Reproduction: the original script rerun and
  `python -m src.stereo.run_stereo --scene Adirondack --legacy-positive-only`
  both gave **1.238896 px / 83.027347%** (294,679 / 354,918).
  Metrics, disparity/depth arrays and the invalid mask matched exactly.
- Formal run: `python -m src.stereo.run_stereo`. With the same matcher
  settings, the protocol includes finite non-sentinel zero disparities.
  Adirondack includes 976 additional pixels on valid GT and gives
  **1.295098 px / 83.302340%** (295,655 / 354,918). Metric depth still
  requires a positive `d + doffs`.
- Acceptance: **15/15 scenes**. Independently recomputed masks, metric depth,
  MAE, coverage, and aggregates from the saved raw disparities, GT and
  calibration. The [per-scene table](../results/tables/middlebury_stereo.csv)
  reports **1.974327 px MAE, 75.980801% coverage, 22.131320 ms** as arithmetic
  means across scenes. Summed matcher time was **331.97 ms**. Each timing is
  one CPU `matcher.compute` call for a newly constructed matcher, with no
  explicit warm-up. Construction, image I/O, grayscale conversion, evaluation,
  plotting, and output writes are outside the timer.
- Illumination comparison (Piano / PianoL): The decoded left images match
  pixel for pixel, and the right images differ in illumination. The left-view
  ground-truth PFM files are byte-identical. Disparity MAE rises from
  **1.2487 px** to **6.1363 px**, while valid coverage falls from **81.85%**
  to **60.85%**. This pair shows that the current baseline is sensitive to
  illumination differences between views. Each MAE uses its scene's own valid
  overlap of predictions and ground truth. One CPU timing per scene is
  insufficient to infer an effect of illumination on runtime.
  [PianoL result](../results/figures/middlebury_pianol.png)
- Outputs: `outputs/stereo/` has the full formal results.
  `legacy_reference/` and `legacy_positive/` under that directory contain
  the two comparison runs. The selected figures show
  [Adirondack](../results/figures/middlebury_adirondack.png) and
  [PianoL](../results/figures/middlebury_pianol.png), the latter having the
  highest MAE (6.1363 px at 60.85% coverage).

## 2026-11-11 — Frame-wise Depth Anything V2-Small (CPU)

- Work is on local branch `part1/framewise-depth`, created directly from
  verified commit `fdb1cba3979a716a316e8854088d1df71f8199ff`.
- The assignment's linked davis_rollercoaster.mp4 has 70 frames, 960 x 540
  pixels at 24 fps. OpenCV metadata and a complete sequential decode agree.
  The requested 100-frame input check rejected the missing frame 70.
  The user explicitly approved processing all 70 source frames, indexed 0..69.
  The optional 195-frame clip is outside this evaluation scope. Reuse the
  verified source clip and frames 0..69 at 24 fps for the Part 2 comparison.
- The official model code is pinned to
  `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` and its working tree is clean.
  The relative-depth vits model uses features=64 and channels=[48,96,192,384].
  Model weights are loaded strictly with the official architecture.
- The checkpoint revision is `03876f8651c73a60fe4c2c48294e09fcb6838fcf`.
  Its SHA-256 is `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`.
  The video source commit is `4f5ae23172ba60fd7bc11ef671cca678842c7072`.
  Video SHA-256 is `7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8`.
  Pinned URLs and verification logic are in `scripts/prepare_framewise.py`.
- Execution used the Windows `cvproject3` environment, Intel Core i5-1135G7,
  CPU float32, two PyTorch threads and one OpenCV thread. CUDA is unavailable.
  Python 3.10.22, torch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 2.2.6,
  opencv-python 5.0.0.93, Matplotlib 3.10.9 and PyYAML 6.0.3 were verified.
- Official preprocessing was preserved at input-size setting 518. Actual
  input tensor shape was 1 x 3 x 518 x 924. Predictions were interpolated to
  native height 540 and width 960 and saved as float32.
- The first 5 frames passed shape, finiteness, source pixel hash, numbering,
  timestamps and video checks. After a separate warm-up, mean inference
  was 2.476270 s/frame, estimating
  247.63 s for 100 frames.
- The full run completed **70/70** frames on CPU. Mean inference was
  **4.741843 s/frame**, median
  **3.008672 s/frame**, and summed inference
  **331.93 s**. Model-load-through-export wall time
  was 340.38 s. Source preflight and Python import time
  are outside that wall-time value.
- Frame 28 took 114.008 s.
  This outlier is retained in the mean and raw timing log. Its cause remains
  undetermined. The full-run mean extrapolates to
  474.18 s for 100 frames.
- Per-frame timing includes official preprocessing, device transfer, forward,
  source-resolution interpolation and CPU copy. It excludes decoding,
  output writes and rendering. The warm-up is recorded separately.
  Peak GPU memory is inapplicable to this CPU execution.
- The exported RGB/depth video has 70 frames at source rate 24 fps and size
  1920 x 588, including a 48-pixel caption row. OpenCV TURBO uses one
  **[0.000000, 3.818236]** range across
  the entire 70-frame run. The pilot has its own 5-frame range.
- Final acceptance used the existing artifacts. The 15-row stereo summary
  plus its equal-weight mean matches the saved stereo metrics CSV. For the
  video, all 70 source hashes, frame indices, timestamps, native float32
  predictions, finite values and extrema agree with `frames.csv`, `run.json`
  and `acceptance.json`. The first five raw predictions match the pilot.
  A temporary MP4 reconstructed from the source frames, saved predictions
  and global TURBO range matched all 70 decoded frames of the existing MP4
  pixel for pixel. The export decodes at 24 fps. Visual inspection sampled
  decoded frames 0, 35 and 69. The [selected stills](../results/figures/framewise_dav2_small_rollercoaster.jpg)
  show those three frames. Inference was not rerun.
- Full evidence is under `outputs/framewise/dav2_small_70/`, including
  `run.json`, `frames.csv`, `frame_indices.npy`, `acceptance.json`,
  `predictions/`, `rgb_depth.mp4` and three preview PNGs.
  The pilot is under `outputs/framewise/dav2_small_pilot5/`.
- Reproduce with `python scripts/prepare_framewise.py`, then
  `python -m src.monocular.run_framewise --frames 70 --device cpu --cpu-threads 2 --output-dir outputs/framewise/dav2_small_70`
  using a fresh output directory. The [frame-wise guide](../src/monocular/README.md)
  provides pinned installation and GPU/Colab handoff commands.
- GPU/Colab execution and a 100-distinct-frame run remain unexecuted.

## 2026-11-12 — NYU Depth V2 UniDepthV2 ViT-S (CPU)

- Branch `part2/nyu-metric-depth` starts from the unmerged Part 1 tip
  `0fe1fde4ae88dcc999683423149cafa524f4c765`.
- NYU labeled MAT SHA-256 is `2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa`. The official download
  was slow, so the original MAT was recovered from the pinned mirror.
  Full size and mirror SHA-256 passed. The first MiB and three 4 KiB
  samples from the middle and tail matched the NYU server. The official
  split SHA-256 is `6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`.
- The split contains 795 development and 654 test images with no overlap.
  MATLAB indices were converted to zero-based Python indices. The HDF5
  loader reads one image and filled-depth map at a time. Three development
  previews (indices 2, 10 and 23) were visually checked for RGB/depth
  orientation and crop alignment.
- Official UniDepthV2 ViT-S revision `038c238f06c87b6c2f5b3749fd51fbf442b1f218` uses
  upstream code `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`. The configuration and
  safetensors checksums are recorded in each `run.json`. The model receives
  only RGB, uses the official default preprocessing, and runs in float32
  on an i5-1135G7 CPU with four PyTorch threads. The independent
  `.venv/nyu-metric` environment uses Python 3.10.22,
  torch 2.6.0+cpu, torchvision 0.21.0+cpu
  and h5py 3.14.0.
- The 15-image development pilot took 63.44 s including
  load and warm-up. Warmed inference averaged 2.6248
  s/image with a 2.5737 s median. Peak process
  working set was 899.2 MiB and minimum sampled
  system available memory was 3443.9 MiB.
  End-to-end per-image timing estimated 29.3
  min for 654 images, excluding setup. All 15 saved predictions and
  metrics passed independent recomputation. These are development checks.
- The formal run completed **654/654** images in **32.62
  min** from model loading through prediction writes. Warmed inference
  averaged **2.8982 s/image**, median
  **2.9188 s/image**. Peak process working
  set was **900.4 MiB**, with minimum sampled
  available system memory of **3231.3 MiB**.
  The per-image timer includes HDF5 read, inference, metrics and prediction
  writes. The inference-only timer covers tensor conversion, official
  inference, validation and any resize. One dev RGB warm-up is timed separately.
- Formal equal-weight means are **AbsRel 0.114609**,
  **RMSE 0.398949 m**, and **δ1 0.892103**.
  The protocol uses 480 × 640 predictions, crop `[45:471, 41:601]`,
  finite ground truth in `[0.001, 10]` m, and prediction clipping to the
  same value bounds. Test labels are used for scoring only, with no
  scale alignment. All 654 saved predictions were validated, and independently
  recomputed metrics matched the CSV and final aggregates.
- An exact float32 comparison in crop `[45:471, 41:601]` across all 654
  test images found zero pixels and zero images at each endpoint,
  `0.001 m` and `10 m`. The inclusive and exclusive ranges produce the
  same ground-truth masks, so the reported metrics are unchanged.
- Full results are local under `outputs/nyu/unidepth_v2_vits14_test654/`. The
  `per_image.csv`, `run.json`, `acceptance.json` and full-size float32
  `predictions/` retain original predictions before cropping and value
  clipping for Part 3. The development run is under `outputs/nyu/unidepth_v2_vits14_dev15/`.
  Use the [NYU guide](../src/metric/README.md) for setup and reproduction.


Copy this template for each experiment:

```text
Run ID / date:
Owner:
Purpose:
Project Git commit / uncommitted changes:
External repository commit:
Model and checkpoint:
Dataset / split / scene or frame IDs:
Hardware / software versions:
Configuration / command:
Precision / input size / output size:
Evaluation mask / aggregation:
Metrics:
Runtime / timing scope / warm-up:
Peak GPU memory:
Raw output path:
Selected figures:
Observations / failure cases / next step:
```
