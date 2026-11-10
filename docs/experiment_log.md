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
