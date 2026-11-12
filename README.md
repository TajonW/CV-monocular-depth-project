# Monocular Metric Depth Estimation and Video Consistency

AIAA 3201 Introduction to Computer Vision Course Project.

## Team

- Tingzhi WANG
- Dongyou LI

See the [collaboration handoff guide](docs/collaboration_guide.md) for publishing
the current baseline branch and sharing selected results. Ready-to-view videos
are in [results/videos](results/videos/).

## Project Tasks

- Part 1: Stereo geometry and frame-wise depth estimation.
- Part 2: Metric depth evaluation and video depth comparison.
- Part 3: One lightweight improvement and a depth-aware application.

## Environment Setup

The classical stereo experiment is implemented and evaluated on all 15 scenes.
Frame-wise Depth Anything V2-Small inference is implemented and all 70 frames
of the supplied clip have been processed and verified on CUDA. The course's
100-frame input discrepancy remains unresolved. Video Depth Anything-Small
has also processed and verified all 70 available frames in offline FP16 mode
with input_size 518. A synchronized three-column RGB/DA V2/VDA video is
generated and verified. RGB-only UniDepthV2 ViT-S has also completed all 654
NYU test images: AbsRel 0.114546, RMSE 0.398888 m, delta1 0.892217, without
test-scale fitting. See [the metric guide](src/metric/README.md) and
[the video guide](src/video/README.md). Case-based video analysis is documented
in [the qualitative analysis](docs/video_qualitative_analysis.md). Part 3 is pending.

For the completed CPU stereo experiment, use the verified direct dependencies:

```powershell
python -m pip install -r requirements-stereo.txt
```

See [the stereo reproduction guide](src/stereo/README.md).

Use a dedicated Python environment (Python 3.11 is the proposed starting point).
`requirements.txt` covers stereo processing, frame-wise Depth Anything V2,
Video Depth Anything inference/export, and NYU metric evaluation dependencies. It does not install
official model code or download checkpoints.

The local development machine has an NVIDIA GeForce RTX 4070 Laptop GPU
(8188 MiB VRAM), driver 560.94, and driver-reported CUDA support up to 12.6.
The selected initial pair is PyTorch 2.7.1 / torchvision 0.22.1, using CUDA 12.6
builds from the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v271).
Inside the activated project environment, install PyTorch first, then the other
requirements:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Other machines should select the CPU/CUDA build appropriate to their hardware
and drivers, keeping the matching package versions. CUDA wheel selection is
documented here rather than forced as a global index in requirements.txt.

Check the installed versions and GPU availability:

```powershell
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

The CPU stereo dependencies have been verified and pinned separately in
`requirements-stereo.txt`. The verified frame-wise direct dependencies are pinned
in `requirements-framewise.txt`; full-clip CUDA FP32 inference is verified.
Video Depth Anything's full 70-frame FP16 run and UniDepthV2's complete NYU
test654 evaluation are also verified. The metric dependency set is pinned in
`requirements-metric.txt`. The optional Gradio web demo is not included.

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
The official `splits.mat` is prepared in `data/nyu/` and validated as the
assignment's 795 development / 654 test split. UniDepthV2 ViT-S has completed
the 654-image test evaluation. See [setup, results and reproduction](src/metric/README.md).

### Video and Model Weights

The required clip is `assets/example_videos/davis_rollercoaster.mp4` from the
Video Depth Anything repository. Place a copy in `data/videos/` and use its first
100 frames at the original frame rate.

Input check: the current repository clip and its local copy both decode to
**70 frames at 24 FPS**. Confirm the course input/protocol before a 100-frame run.
The runner checks the decoded prefix strictly and never duplicates frames to
satisfy the requested count. Single-frame validation is unaffected.

Download model checkpoints following each official repository's instructions.
Keep weights local; document exact model variants and checkpoint paths when
inference is implemented.

## How to Run

From the repository root in the activated environment:

```powershell
python -m unittest discover -s tests -v
python -m src.stereo.run_stereo --run-name my_baseline
python -m src.stereo.evaluate_stereo --run-dir outputs/stereo/my_baseline
```

Use a fresh run name: completed runs are not overwritten. To run only one sample,
append `--scenes Adirondack`. See `src/stereo/README.md` for sensitivity experiments
and selected result generation.

For first-frame Depth Anything V2-Small validation:

```powershell
python -m src.monocular.run_framewise --run-name first_frame_check --max-frames 1
python scripts/verify_framewise_run.py --run-dir outputs/framewise/first_frame_check
```

See [frame-wise setup and outputs](src/monocular/README.md). The delivered
`first_frame` run already exists; use a fresh name for reruns.

To reproduce the full available 70-frame clip and check all saved outputs:

```powershell
python -m src.monocular.run_framewise --run-name rollercoaster_70_repro --max-frames 70
python scripts/verify_framewise_clip.py --run-dir outputs/framewise/rollercoaster_70_repro
```

The delivered run is `outputs/framewise/rollercoaster_70/`. Its original MP4
and an additional H.264-compatible export both contain 70 frames at 24 FPS.

## Results

Classical stereo: equal scene averages over 15 scenes give **MAE 2.033 px**,
**coverage 75.86%**, and **38.25 ms per pair** (matcher only, CPU single thread;
median of 3 calls after one warm-up per scene). Pixel-weighted MAE is 1.906 px.
The original window-5 baseline is retained; window-3/9 sensitivity runs are also
saved. See `results/tables/stereo/` and `results/figures/stereo/`.

The optional explicit left-right consistency diagnostic is also complete:

```powershell
python -m src.stereo.diagnose_left_right
```

At a fixed 1 px threshold, retained predictions have MAE 1.513 px and coverage
63.72% (equal scene means). This is a different pixel subset, not a claim that
the full depth map became more accurate. The original baseline is preserved.
See [diagnostic definitions](docs/left_right_diagnostic.md).

Keep complete floating-point predictions in the ignored `outputs/` directory.

Frame-wise validation: frame 0 produces a finite float32 relative-depth map at
960 x 540, from a 924 x 518 network input. The saved FP32 result exactly matches
the official model interface. This does not establish temporal stability or
metric accuracy. See `results/figures/framewise/first_frame_preview.png` and
`results/tables/framewise/first_frame_verification.json`.

Full available-clip run: all **70 frames**, native **24 FPS**, **2.917 s** duration;
one global min/max of **0 to 3.818238** across the full clip. Mean inference time
is **54.86 ms/frame** and mean preprocessing/inference/transfer time is
**79.03 ms/frame**, excluding decoding and file writes. PyTorch peak allocated
GPU memory is **413.57 MiB**, reserved **574 MiB**. These are relative-depth
predictions without video ground truth. See selected frames in
`results/figures/framewise/rollercoaster_70_contact_sheet.png` and all-frame
checks in `results/tables/framewise/rollercoaster_70_clip_verification.json`.

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

