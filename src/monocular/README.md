# Part 1: Frame-wise relative depth

Independent Depth Anything V2-Small inference is implemented. Frame 0 and all
70 frames of the supplied video are verified on CUDA in FP32. The source has
fewer than the 100 frames requested by the course; the completed local run is
explicitly a 70-frame experiment, with no metric accuracy claim.

## Setup and run

Use the activated `cv_project` environment (Python 3.11). The tested direct
dependencies are in `requirements-framewise.txt`; install the matching PyTorch
build first, as documented in the root README. Gradio and xFormers are not
required for the verified official fallback attention path.

Required local files:

- `external/Depth-Anything-V2/`: official model code.
- `checkpoints/depth_anything_v2_vits.pth`: relative-depth Small weights.
- `data/videos/davis_rollercoaster.mp4`: specified input clip.

From the project root:

```powershell
python -m src.monocular.run_framewise --run-name first_frame_check --max-frames 1
python scripts/verify_framewise_run.py --run-dir outputs/framewise/first_frame_check
```

The delivered run is `outputs/framewise/first_frame/`. Choose a fresh name on
reruns. `configs/framewise.yaml` defaults to one frame, Small/vits, input size
518, auto device, FP32 and one warm-up. CLI options can override `--input-size`,
`--device` (auto/cpu/cuda), and `--precision` (fp32/fp16). FP16 requires CUDA;
only CUDA FP32 was validated in this task.

The completed full available-clip run is `outputs/framewise/rollercoaster_70/`.
To reproduce without overwriting delivered results:

```powershell
python -m src.monocular.run_framewise --run-name rollercoaster_70_repro --max-frames 70
python scripts/verify_framewise_clip.py --run-dir outputs/framewise/rollercoaster_70_repro
```

The model uses the official transforms: BGR-to-RGB/255, aspect-preserving
lower-bound resize to multiples of 14, cubic interpolation, ImageNet
normalization and CHW float32 layout. Explicit device placement supports CPU
mode without the official helper's automatic CUDA selection. Output is resized
to native resolution with bilinear align_corners=True. The verified frame-0
input tensor and raw prediction match the official interface exactly.

## Saved output

```text
outputs/framewise/<run-name>/
  config.yaml                 Effective configuration
  manifest.json               Commits, versions, input/checkpoint/code SHA256
  summary.json                Device, shapes, timings, memory and scope
  environment-freeze.txt       Full installed environment
  metrics.csv                 Per-frame timings and raw statistics
  normalization.json          Fixed range over the processed prefix
  preview.png                 First RGB/relative-depth comparison
  verification.json           Produced by verify_framewise_run.py
  rgb_relative_depth.mp4       Multi-frame original MP4 (mp4v)
  rgb_relative_depth_h264.mp4  Optional H.264 export for common players
  h264_encoding.json          Exact external encoder command and provenance
  clip_verification.json      All-frame/video checks from verify_framewise_clip.py
  frames/000000/
    rgb.png                   Lossless decoded first frame
    relative_depth.npy        Native-size float32, NOT meters
    relative_depth.png        Color visualization
    comparison.png            Labeled side-by-side comparison
    metadata.json             Index, timestamp, dimensions and timings
```

Selected previews/JSON records are also copied to `results/figures/framewise/`
and `results/tables/framewise/`. Full raw arrays stay in ignored outputs.

Each prediction is independent: no temporal state, fusion, scale alignment or
ground-truth calibration. All requested raw maps are saved first; visualization
then uses a single global min/max for that processed prefix. The first-frame
range is only for this preview and must not replace full-clip normalization.
Multi-frame runs write an MP4 at the original FPS and check encoded FPS/frame
count. All 70 available frames have now been verified: raw RGB order, float32
finite predictions, shared normalization, exact color/comparison PNGs, and
decoded frame count/FPS for both MP4 versions. The saved frame-0 raw prediction
also exactly matches the earlier official-interface-verified first-frame run.

### Delivered clip

- 70 frames (indices 0-69), 24 FPS, 2.916667 seconds, 1932x588 comparison canvas.
- Native depth output 540x960, network input 518x924, FP32 on RTX 4070 Laptop.
- Fixed global range 0-3.818237782, inferno palette; no per-frame normalization.
- Mean forward/output-resize 54.8583 ms, pipeline with transforms/transfers
  79.0347 ms. PyTorch peak allocated 413.57 MiB / reserved 574 MiB.
- All 36,288,000 raw prediction pixels are finite; no duplicated or padded frames.
- Selected frames 0/35/69 are in
  `results/figures/framewise/rollercoaster_70_contact_sheet.png`.

An additional H.264/yuv420p version was exported from the lossless comparison
PNGs, not by recompressing the original MP4. FFmpeg is optional for this export;
it is not required by the inference script. With a libopenh264-enabled FFmpeg
on PATH, the equivalent command is:

```powershell
ffmpeg -n -framerate 24 -start_number 0 -i "outputs/framewise/rollercoaster_70/frames/%06d/comparison.png" -frames:v 70 -fps_mode passthrough -c:v libopenh264 -b:v 8M -pix_fmt yuv420p -g 24 -movflags +faststart -an "outputs/framewise/rollercoaster_70/rgb_relative_depth_h264.mp4"
```

The exact binary version, command and output hash are in `h264_encoding.json`;
the verifier checks this compatible version too when present. The source
video and floating-point depth predictions are never changed by encoding.

CUDA timings are synchronized. `inference_ms` covers forward plus output
resize; `pipeline_ms` adds preprocessing and host/device transfers. Both exclude
decoding, model load, warm-up, visualization and disk I/O. One measured frame is
not a stable speed benchmark. GPU memory is PyTorch peak allocated/reserved
memory after warm-up, including model weights, not total process/OS usage.

## Source length and tests

The current local clip decodes to **70 frames at 24 FPS**, byte-identical to the
video tracked in the cloned Video Depth Anything repository. The course asks
for 100 frames. A request for 100 fails during preflight decoding before model
loading or output creation: no looping, padding, resampling or silent truncation.
The user confirmed no additional course video is currently available. The
70-frame local experiment is complete, while acceptance of fewer than 100 frames
still needs course clarification. First-frame history: `docs/framewise_first_frame.md`.

```powershell
python -m unittest discover -s tests -v
```

Tests cover global normalization, fixed-value colors, constant/nonfinite maps
and strict prefix length, alongside stereo tests. The separate real-model
verification above checks the source/checkpoint hashes, original RGB, official
preprocessing and prediction, and saved color visualization.
