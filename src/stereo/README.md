# Part 1: Stereo geometry

This module implements and evaluates the completed 15-scene CPU StereoSGBM
baseline. Commands below run from the repository root in Python 3.11.

```powershell
conda activate cv_project
python -m pip install -r requirements-stereo.txt
python scripts/prepare_middlebury.py
python -m unittest discover -s tests -v
python -m src.stereo.run_stereo --run-name adirondack_check --scenes Adirondack
python -m src.stereo.run_stereo --run-name my_baseline
python -m src.stereo.evaluate_stereo --run-dir outputs/stereo/my_baseline
```

Completed run directories are never silently overwritten. Choose a fresh name
for reruns. The delivered `baseline`, `window3`, and `window9` runs already exist
locally. On a clean clone, regenerate them with:

```powershell
python -m src.stereo.run_stereo --run-name baseline
python -m src.stereo.run_stereo --run-name window3 --block-size 3
python -m src.stereo.run_stereo --run-name window9 --block-size 9
python scripts/summarize_stereo.py
```

To add the optional standalone left-right diagnostic after the baseline:

```powershell
python -m src.stereo.diagnose_left_right
```

It reads `outputs/stereo/baseline/` and writes a separate
`outputs/stereo/left_right/` folder. Choose a fresh `--output-dir` on reruns.
No extra dependencies are needed. It independently computes signed right-view
disparities and checks `abs(dL + dR(xL-dL,y)) <= 1` with nearest-pixel sampling.
Both directions retain the baseline's internal OpenCV filtering parameters.
This is a standalone diagnostic on top of the existing baseline, not a rerun
of unfiltered SGBM. Original prediction files remain unchanged.

Each scene saves a six-panel `overview.png`, a five-class `status.png` label map
(integer codes 0-4), binary inconsistent/consistent masks, `diagnostic.npz` with
float residuals and right disparities, and resolved right matcher parameters.
Uncheckable locations have NaN residuals. Detailed definitions and caveats:
`docs/left_right_diagnostic.md`. Selected figures/statistics are copied into
`results/figures/stereo/` and `results/tables/stereo/` for the final project report.

The sensitivity runs change block size and its formula-derived P1/P2 together;
they are not an independent Part 3 improvement or a held-out tuning experiment.

- Read im0.png, im1.png, calib.txt, and disp0GT.pfm.
- Compute OpenCV StereoSGBM disparity and divide the fixed-point result by 16.
- Use quarter-resolution calibration without scaling it again.
- Convert to meters: Z = f * baseline / (disparity + doffs) / 1000.
- Keep an explicit invalid prediction mask and floating-point outputs.
- Report disparity MAE, coverage, and runtime.

See `docs/evaluation_protocol.md` for mask and aggregation rules.

## Saved artifacts

Each run in `outputs/stereo/<run-name>/` includes:

- `config.yaml`, `metrics.csv`, `summary.json`.
- `manifest.json`: hardware, versions, command, input SHA256 and source hashes,
  parent Git revision and dirty-worktree state.
- `environment-freeze.txt`: the full installed environment; it also includes
  neural-model packages present on the machine, which CPU stereo does not need.
- `<scene>/prediction.npz`: raw int16 fixed-point disparity, float32 pixel
  disparity, float32 metric depth and separate boolean validity masks.
- `<scene>/metrics.json`, `resolved_parameters.json`, `invalid_mask.png`,
  `overview.png`. Gray plot backgrounds indicate invalid/unavailable values.

Use `numpy.load` to read prediction.npz. Invalid float pixels are NaN.
Ground truth remains separate in the local dataset and never enters matching.
Runtime measures `matcher.compute` only, not end-to-end image processing.

The PFM loader handles bottom-up rows and byte order, preserving pixel values
regardless of the display-scale magnitude in Middlebury headers. All 15 files
were cross-checked against torchvision's reader. No extra `mask0nocc.png` mask
is applied: the course evaluation uses all finite positive GT disparities.
