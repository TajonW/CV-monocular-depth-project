# Part 2: Video depth

Official Video Depth Anything-Small offline relative-depth inference has passed
a 32-frame CUDA smoke test with FP16 autocast at input_size 518 and 364. These
are prefix checks, not a completed 70-frame comparison or accuracy evaluation.
The complete available 70-frame clip has subsequently been processed and
verified with input_size 518. A synchronized three-column model-comparison
video is generated and verified. Case-based qualitative analysis is recorded in
[the analysis notes](../../docs/video_qualitative_analysis.md).

## Reproduce

Activate `cv_project` (Python 3.11). Install the matching CUDA PyTorch build as
described in the root README, then `python -m pip install -r requirements.txt`.
Required local inputs are the official repository at
`external/Video-Depth-Anything/`, Small relative checkpoint at
`checkpoints/video_depth_anything_vits.pth`, and the existing
`data/videos/davis_rollercoaster.mp4`.

From the project root, use a fresh output name:

```powershell
python -m src.video.run_video --run-name smoke_32_364_repro --max-frames 32 --input-size 364
python scripts/verify_video_run.py --run-dir outputs/video/smoke_32_364_repro
```

`configs/video.yaml` defaults to 32 real frames, Small/vits, offline FP16
autocast, max_res 640 and input_size 364. The same prefix at input_size 518
also passed, but its memory use was much higher. xFormers is not installed;
official fallback attention was used. Decoding uses the official OpenCV path.

## Delivered GPU checks

Both runs preserve indices 0-31 and 24 FPS: 1.333 seconds. The original
960x540 RGB is resized by the official reader to 640x360; raw float32 depth
has shape (32, 360, 640). Both Small checkpoints loaded strictly.

| Setting | Network window shape | Peak allocated MiB | Peak reserved MiB | Cold clip pipeline s |
|---|---|---:|---:|---:|
| input_size 518 | Not captured in initial log; official resize implies 1x32x3x518x924 | 8014.22 | 11858 | 33.564 |
| input_size 364 | Captured: 1x32x3x364x644 | 2385.35 | 3162 | 2.232 |

The initial 518 run's standard module hooks were bypassed by the official
direct `self.forward` call. Clip timing and output checks remain valid; no
per-window timings are claimed for that run. The observer was corrected for
the 364 run, which captured and checked both windows. Source snapshots preserve
each executed version. Reserved memory is PyTorch allocator accounting, not a
measurement of dedicated physical VRAM; its high value does not prove a
particular OS paging mechanism. The 364 setting provides more memory headroom.

These are cold smoke timings without warm-up. Clip pipeline time includes
official preprocessing, CUDA inference, transfers, output resize and overlap
alignment; it excludes decode, initialization, export and file writes. It is
not an isolated model latency or a steady-state benchmark. Lower input size
can change prediction quality and needs the full-clip qualitative comparison.

## Compatibility and verification

`visualization.py` replaces the removed `cm.get_cmap` lookup with
`matplotlib.colormaps['inferno'].colors`, retaining the official integer
`floor(normalized * 255)` color lookup. This matches the original formula
exactly over all saved predictions. It also handles constant maps without
division by zero. The external repository, model weights and temporal inference
are unmodified. Videos use H.264 via imageio's bundled FFmpeg.

Official inference fixes window length 32, overlap 10 and step 22. A 32-frame
prefix executes two windows, internally repeating the final frame for context;
it returns exactly 32 real-frame outputs. The model's original overlap
scale/shift alignment and blending are preserved and use no ground truth.

Every run stores config, manifest (versions, input/checkpoint/code hashes),
environment freeze, project source snapshots, `depths.npz`, `input_rgb.npz`,
one whole-prefix normalization, a contact sheet, depth MP4 and RGB/depth MP4.
The independent saved-output verifier checks source RGB order, all raw arrays,
code/input/checkpoint hashes, exact color formula and every encoded frame's
content, count, size and FPS. Selected records are in `results/tables/video/`;
contact sheets are in `results/figures/video/`.

## Complete available clip and next comparison

The delivered full run is `outputs/video/rollercoaster_70_518/`. It processed
indices 0-69 from scratch with input_size 518, max_res 640, offline FP16
autocast, four 1x32x3x518x924 windows and original 24 FPS. All 70 raw float32
maps are finite and nonnegative, at 640x360. The source ends after frame 69.

```powershell
python -m src.video.run_video --run-name rollercoaster_70_518_repro --max-frames 70 --input-size 518
python scripts/verify_video_run.py --run-dir outputs/video/rollercoaster_70_518_repro
```

The cold synchronized pipeline took 55.3061 s (790.087 ms per real output
frame, including overlap work). No CUDA OOM or inference errors occurred.
PyTorch peak allocated was 8014.219 MiB and reserved was 11858 MiB. Physical
GPU memory reported by PyTorch is 8187.5 MiB; allocator reservation is not
dedicated VRAM consumption. A live nvidia-smi sample showed 7923 MiB used.
The videos contain exactly 70 frames at 24 FPS (2.916667 s), with one fixed
inferno range [0, 7.433829784] across the whole run. The independent verifier
checked all 16,128,000 raw values and every frame of both H.264 videos.

No prefix results or prefix normalization were reused: later context can
change endpoint predictions. At comparison time, use exactly the same real
frames and palette for DA V2-Small and VDA-Small, fixed whole-clip normalization
per method, a common display size, and disclose both models' actual input sizes
and precision. The source has only 70 frames although the course asks for 100;
course clarification is still needed. NYU metric evaluation is separate.

## Three-column comparison video

The delivered comparison is
`outputs/comparison/rollercoaster_70/rgb_da2_vda_comparison.mp4`:
left RGB, center Depth Anything V2-Small, right Video Depth Anything-Small.
It contains 70 frames at 24 FPS (2.916667 seconds), H.264, canvas 1944x488.
Headers disclose frame index, timestamp, method and precision.

Reproduce from the saved floating-point predictions, without GPU inference:

```powershell
python -m src.video.compare_models --output-dir outputs/comparison/rollercoaster_70_repro
python -m src.video.compare_models --verify-only --output-dir outputs/comparison/rollercoaster_70_repro
```

Each panel is 640x360. The display-only resize operates on scalar depth with
OpenCV INTER_LINEAR before applying the same inferno/255-index lookup to both
methods. Native predictions remain unchanged. Each method retains its own
original full-native-clip normalization: DA V2 [0,3.818237782], VDA
[0,7.433829784]. Colors therefore do not encode a shared metric distance.
There is no per-frame normalization, temporal filtering or fitted cross-model
scale. Original model settings remain DA V2 FP32 and VDA FP16 autocast, with
network spatial input 924x518 for both, and native depth output 960x540 and
640x360 respectively.

Generation automatically verifies source RGB/frame order for both models,
native raw bounds, all lossless comparison PNGs, and every encoded frame's
content, count, size and FPS. The manifest hashes original inputs, model-run
records and a saved source snapshot. Selected frames 0/35/69 are in
`results/figures/comparison/rollercoaster_70_contact_sheet.png`; provenance and
checks are in `results/tables/comparison/`. The video supports qualitative
analysis but does not establish metric accuracy or improved temporal quality
without reviewing the actual motion and failure cases.

## Qualitative analysis and evidence

```powershell
python scripts/analyze_video.py
```

The case-based review finds that both methods retain major track/support
structure. A tree-region RGB tracking diagnostic over frames 64-69 supports a
local stability benefit for VDA under the existing per-method display ranges;
it is not a GT temporal-error score or an accuracy result. Thin structures,
vehicle details and low-contrast distant foliage retain shared limitations.
The selected motion/edge cases do not justify an all-region superiority claim.

Evidence strips and a tracked-feature plot are in `results/figures/video_analysis/`;
ROI definitions, all RGB tracks and provenance are in `results/tables/video_analysis/`.
The optional `outputs/analysis/video_qualitative/comparison_slow_4x.mp4` is a
6-FPS review copy, explicitly labeled as 4x slower; source timestamps and all
70 real frames are retained. The original 24-FPS comparison is unchanged.
The notes include measured settings/costs, limits on interpretation and a
report-ready English paragraph. No model or stored prediction was modified.
