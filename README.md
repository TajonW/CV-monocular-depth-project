# Monocular Metric Depth Estimation and Video Consistency

AIAA 3201 Introduction to Computer Vision Course Project.

## Team

- Tingzhi WANG
- Dongyou LI

## Project Tasks

- Part 1: Stereo geometry and frame-wise depth estimation.
- Part 2: Metric depth evaluation and video depth comparison.
- Part 3: One lightweight improvement and a depth-aware application.

## Environment Setup

The repository currently contains the project structure and prepared Middlebury
data. Inference and evaluation scripts have not been implemented yet.

Use a dedicated Python environment. The initial CPU stereo dependencies are:

```powershell
python -m pip install -r requirements.txt
```

Run this command inside your project environment. PyTorch and model-specific
dependencies will be added when the inference environment is verified.
Dependency versions are not locked yet.

## Repository Structure

```text
configs/                   Experiment settings
scripts/                   Data preparation utilities
src/stereo/                Part 1: stereo matching and evaluation
src/monocular/             Part 1: frame-wise relative depth
src/metric/                Part 2: metric depth and NYU evaluation
src/video/                 Part 2: video depth inference
src/improvement/           Part 3: the selected improvement
src/application/           Depth-aware application
src/utils/                 Shared data I/O and visualization
results/tables/            Selected metric tables for GitHub
results/figures/           Selected comparison and failure figures
docs/                      Evaluation protocol and experiment log
report/                    CVPR report sources and final PDF
```

Local-only directories (ignored by Git):

```text
data/downloads/            Original dataset archives
data/middlebury/MiddEval3/ Extracted quarter-resolution stereo data
data/nyu/                 NYU labeled data and split file
data/videos/              Required input clip
checkpoints/              Pretrained model weights
external/                 Clones of official model repositories
outputs/                  Full predictions, videos, and intermediate results
docs/assignment/          Original course project instructions
tmp/                      Temporary files
```

## Data Preparation

Paths below are relative to the repository root.

### Middlebury

Original archives are kept in `data/downloads/`. Both archives are extracted
into `data/middlebury/`, giving 15 required scene folders under
`data/middlebury/MiddEval3/trainingQ/`.

Each required scene has `im0.png`, `im1.png`, `calib.txt`, and `disp0GT.pfm`.
The actual archive directory is named **trainingQ**, although the assignment
calls this subset trainQ. The archives' `testQ` subset is not extracted.

To prepare the same layout on another machine, put the two archives in
`data/downloads/` and run (Python 3.10+; standard library only):

```powershell
python scripts/prepare_middlebury.py
```

The script checks that all 15 scenes have the four required files.

### NYU Depth V2

The labeled file is stored at `data/nyu/nyu_depth_v2_labeled.mat`.
The official `splits.mat` must still be downloaded into `data/nyu/`.
Use the assignment's 795 development / 654 test split.

### Video and Model Weights

The required clip is `assets/example_videos/davis_rollercoaster.mp4` from the
Video Depth Anything repository. Place a copy in `data/videos/` and use its first
100 frames at the original frame rate.

Download model checkpoints following each official repository's instructions.
Keep weights local; document exact model variants and checkpoint paths when
inference is implemented.

## How to Run

Runnable project commands will be added alongside the corresponding scripts.
The current directories contain implementation notes, not completed algorithms.

Start with Part 1: load the Adirondack stereo pair, compute disparity, convert
it to metric depth, and evaluate against its PFM ground truth. Then process all
15 trainingQ scenes and implement the frame-wise video baseline.

## Results

No experiments have been run yet. Keep complete predictions in `outputs/` and
copy only selected figures and tables into `results/`.

See [the evaluation protocol](docs/evaluation_protocol.md) and
[the experiment log](docs/experiment_log.md) before recording results.

## Submission

- Public repository with reproducible setup, data preparation, and commands.
- A 6-8 page CVPR-format PDF report, excluding references.
- `depth_results.zip` containing the required RGB/depth video comparisons and
  a simple depth-aware application output; submit this separately.

The original assignment is kept locally in `docs/assignment/`.

## References and External Code

Record the exact commit and checkpoint for each external model used.

- [OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
- [Middlebury Stereo](https://vision.middlebury.edu/stereo/)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)

The full report bibliography will include the assigned readings, datasets,
models, and any borrowed implementation components.

