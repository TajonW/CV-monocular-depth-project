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

The CPU Middlebury stereo baseline is implemented and evaluated on all 15
trainingQ scenes. The dataset and full outputs are local-only.

Use a dedicated Python environment. The initial CPU stereo dependencies are:

```powershell
python -m pip install -r requirements.txt
```

Run this command inside your project environment. The tested frame-wise model
versions are pinned in `requirements-framewise.txt`; CPU and GPU installation
commands are in [the frame-wise guide](src/monocular/README.md).

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

Place the 15 scene folders under `data/middlebury/MiddEval3/trainingQ/`.
Reuse an existing verified trainingQ copy if available. The source archives
can be stored in `data/downloads/`.

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

Run `python scripts/prepare_framewise.py` to prepare the pinned upstream code,
Depth Anything V2-Small weights and the assigned `davis_rollercoaster.mp4`.
The script checks the video SHA-256
`7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8` and
the checkpoint hash, and pins the model code to
`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. Assets are stored locally under
`data/videos/`, `checkpoints/` and `external/`.

The assigned clip contains 70 source frames at 24 fps. Part 1 uses all frames
`0–69` at that rate. The optional 195-frame clip is outside the Part 1
evaluation scope.

## How to Run

### Part 1 stereo baseline (CPU)

Run from the repository root with the Middlebury inputs in place. In the
existing `cvproject3` environment:

```powershell
conda activate cvproject3
python -m pip install -r requirements.txt
# Reproduce the old single-scene result (1.2389 px MAE, 83.03% coverage).
python -m src.stereo.run_stereo --scene Adirondack --legacy-positive-only
# Run all 15 scenes with the course protocol.
python -m src.stereo.run_stereo
```

The runner loads `configs/stereo.yaml` and writes the batch table and scene
outputs to `outputs/stereo/`. The legacy comparison writes separately to
`outputs/stereo/legacy_positive/`. The measured Python and package versions
are in the [experiment log](docs/experiment_log.md).

### Part 1 frame-wise relative depth

The entry point uses Depth Anything V2-Small, with the official preprocessing,
native-resolution float32 predictions and one visualization range over a run.
The assignment specifies the first 100 frames. The linked clip contains 70 frames, so we processed all 70 available frames (0–69) at the original 24 fps.

```powershell
conda activate cvproject3
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-framewise.txt
python scripts/prepare_framewise.py
python -m src.monocular.run_framewise --inspect-only --frames 70
python -m src.monocular.run_framewise --frames 5 --device cpu --cpu-threads 2 --output-dir outputs/framewise/dav2_small_pilot5
python -m src.monocular.run_framewise --frames 70 --device cpu --cpu-threads 2 --output-dir outputs/framewise/dav2_small_70
```

Use a fresh output directory for repeated runs. The reference run is under
`outputs/framewise/dav2_small_70/`, with raw predictions, frame metadata and
`rgb_depth.mp4`. The [frame-wise guide](src/monocular/README.md) includes
GPU/Colab commands that have not been run. The
[experiment log](docs/experiment_log.md) records the actual CPU run and its
timing scope.

## Results

The 15-scene CPU stereo run averaged **1.9743 px MAE**, **75.98% coverage**,
and **22.13 ms per pair** with equal weight per scene. Timing covers one
`matcher.compute` call per scene. See the [per-scene table](results/tables/middlebury_stereo.csv)
and the [representative figures](results/README.md).

The frame-wise relative-depth run completed 70/70 frames and exported a
24 fps RGB/depth video. CPU inference averaged **4.742 s/frame** with a median
of **3.009 s/frame**. The mean includes the **114.008 s** timing at frame 28.
Timing covers preprocessing, forward inference, native-size interpolation
and device transfers, with warm-up, decoding, writing and rendering timed
outside inference. See the [selected frames](results/README.md#frame-wise-relative-depth)
and [experiment log](docs/experiment_log.md).

For Part 2 video comparison, reuse the verified source clip and the same
frame indices `0–69` at 24 fps. See the
[evaluation protocol](docs/evaluation_protocol.md).

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
