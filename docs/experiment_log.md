# Experiment log

## 2026-11-11: Case-based video analysis and reproducible evidence

- Added scripts/analyze_video.py using saved predictions only, without any new
  model inference, smoothing, depth edits or per-frame normalization.
- Generated early/middle/late overview strips and consecutive ROI comparisons:
  fine supports0-5, vehicle34-39, junction54-59, tree64-69. All show synchronized
  RGB / DA V2-Small / VDA-Small with original per-method full-clip scales.
- Tree diagnostic: RGB-only Shi-Tomasi corner candidates in frame64 ROI
  [555,315,632,353], pyramidal Lucas-Kanade forward/backward tracking through69,
  FB <1px and LK patch error <20. All18 candidates survived. The displayed
  primary track was selected by RGB corner strength, not depth behavior.
- Primary DA display-value range0.02627-0.06103; VDA0.05587-0.06279.
  17/18 local tracks have smaller VDA display-value range, one is opposite.
  This is a single correlated spatial ROI over6frames, with no GT/pose; it
  supports a local display-stability observation, not global temporal error,
  physical accuracy, or a causal temporal-module ablation.
- Visual findings: both preserve major structure; neither consistently
  resolves thin members/vehicle details; both retain low-contrast tree regions.
  The selected moving-object cases do not demonstrate a clear ghosting benefit.
- Generated a labeled4x-slower H.264 review copy, exactly70frames at6FPS,
  11.6667s. Original source timestamps retained and original24FPS video unmodified.
- Rechecked all input hashes in the original comparison manifest. Saved script
  snapshot/version hashes, ROI definitions, all tracks, plot and evidence strips.
- Analysis: docs/video_qualitative_analysis.md, including bounded conclusions,
  observed run costs with timing/setting caveats, and an English report paragraph.
  Selected artifacts: results/{figures,tables}/video_analysis/; complete review
  artifacts: outputs/analysis/video_qualitative/. Course70-vs-100frame issue remains.

## 2026-11-11: UniDepthV2 ViT-S single-image and complete NYU test654 baseline

- Downloaded splits.mat from the course-linked NYU official source. Verified
  795/654 unique, disjoint indices covering all 1449 labeled images; converted
  MATLAB indices to zero-based Python exactly once.
  Split SHA256: 6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee.
- Prepared official UniDepth source commit 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
  from a pinned archive while the initial Git clone was slow. The clone later
  completed at the same commit; runtime uses external/UniDepth-source/.
- Downloaded official lpiccinelli/unidepth-v2-vits14, revision
  038c238f06c87b6c2f5b3749fd51fbf442b1f218. model.safetensors SHA256:
  93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648.
- Installed verified dependencies in cv_project, preserving torch2.7.1+cu126.
  Exact direct pins in requirements-metric.txt; pip check passed. No compiled
  KNN/training extension needed for official V2 inference or our 2D metrics.
- Added src/metric/{data,engine,metrics,visualization,evaluate_nyu}.py,
  configs/metric.yaml and scripts/{prepare_unidepth,verify_nyu_run}.py.
  Four protocol tests catch scale alignment, crop/GT-mask/clipping errors,
  non-strict delta1 and silent nonfinite prediction removal.
- First official test image (MATLAB1/Python0) passed: AbsRel0.119539964,
  RMSE0.358897547m, delta1 0.996332160. Saved and visually checked RGB,
  filled GT, native prediction and error; image orientation is correct.
- Full run: python -m src.metric.evaluate_nyu --run-name nyu_test654
  --max-images 654. Frozen RGB-only official infer(rgb), no GT camera/depth
  input, default shape bounds, no test-scale fitting. FP16 CUDA autocast,
  FP32 weights, network1x3x490x644, float32 saved metric depth480x640.
- Evaluation: filled depths, crop[45:471,41:601], finite GT strictly between
  .001 and10m, prediction clip[.001,10] for metrics only. Every metric is
  computed per image and averaged equally over all654 images.
- Results: AbsRel0.11454563736611406, RMSE0.39888762120573146m,
  delta1 0.8922168587467723. 540,915 valid-GT prediction pixels clipped above
  10m for evaluation; original arrays preserved. No samples/hard pixels dropped.
- RTX4070 Laptop, two warm-ups: mean synchronized inference pipeline25.840906ms,
  median25.39755ms; peak allocated484.958MiB/reserved602MiB. Timing includes
  tensor preparation, official preprocess/compute/postprocess and CPU depth
  copy, excluding HDF5 IO, metrics, writes, warm-up and model initialization.
- scripts/verify_nyu_run.py independently checked all654 indices, all
  200,908,800 raw values, 156,018,240 valid GT pixels, all hashes/snapshots,
  every CSV row and equal-image macro means. Passed. The first full-run
  prediction exactly matches the separate single-image run (max diff0).
- Saved fixed examples at test orders0/327/653 and post-evaluation high-AbsRel
  cases MATLAB469,88,517. Observations: broad distance overestimation around
  doorways; excessive far-background depth in shelves/interiors; smoothing of
  close clutter and thin chair boundaries. No parameter changes from test data.
- Outputs: outputs/metric/{first_nyu_image,nyu_test654}/. Selected metric tables,
  provenance, verification and six figures are in results/{tables,figures}/metric/.
  This is the course-protocol metric baseline, not exact paper-score reproduction.

## 2026-11-11: Synchronized three-column RGB/DA V2/VDA video

- Added src/video/compare_models.py. Reused the full existing 70-frame runs;
  no model inference, depth modification, temporal filtering or scale fitting.
- Output: outputs/comparison/rollercoaster_70/rgb_da2_vda_comparison.mp4,
  H.264, 1944x488, 70 frames at original 24 FPS (2.916667 seconds).
- Each panel displays 640x360, ordered RGB / Depth Anything V2-Small /
  Video Depth Anything-Small. Display-only bilinear resizing is applied to
  scalar depth before identical inferno/255-index color lookup. Headers record
  frame/time/method/precision; footer explains independent relative scales.
- Preserved full-native-clip ranges: DA V2 [0,3.818237782], VDA
  [0,7.433829784]. No per-frame or shared cross-model normalization.
- Verified original source hash, frame order and RGB equality for both models,
  all raw dtype/shape/finiteness and whole-clip bounds, all 70 lossless panels,
  and all encoded frames' content/size/FPS. Source ends after frame 69.
- Mean lossy RGB-channel error per panel: RGB 2.72015, DA V2 2.34503,
  VDA 2.31832 (0-255 units; encoding checks, not depth accuracy metrics).
- Video SHA256: aad03bfd1a343fd8d45a32bdcbdc86e8ff40f910f93934af11d4b5b2c956f18a.
- Saved source snapshot, input/code hashes, manifest, verification and all
  lossless frames. Contact sheet uses frames 0/35/69. Selected artifacts are
  in results/{figures,tables}/comparison/.
- Qualitative analysis of flicker, edge stability and motion artifacts remains
  to be written; no temporal-quality superiority is claimed from file checks.

## 2026-11-11: Full available 70-frame VDA-Small run at input_size 518

- Ran python -m src.video.run_video --run-name rollercoaster_70_518
  --max-frames 70 --input-size 518, followed by scripts/verify_video_run.py.
- Small relative-depth checkpoint, unchanged official offline algorithm,
  FP16 autocast, max_res 640, original 24 FPS and indices 0-69.
- Four observed network windows of [1,32,3,518,924]; official overlap alignment
  and internal context-only tail padding preserved. Exactly 70 outputs returned.
- Raw depth shape (70,360,640), float32, all 16,128,000 values finite and
  nonnegative. The resized source RGBs match the same original frames in order.
- One full-clip inferno range [0,7.433829784]; no prefix maps/ranges reused.
- Cold synchronized inference pipeline 55.306119 s (790.087417 ms/real frame);
  no warm-up, excludes decode, model loading, visualization and file writes.
  This is not an isolated forward or steady-state latency benchmark.
- No OOM or inference errors. Peak PyTorch allocated 8014.219238 MiB, reserved
  11858 MiB; physical capacity 8187.5 MiB. A during-inference nvidia-smi sample
  read 7923 MiB dedicated memory use and 100% GPU utilization. Reservation is
  allocator accounting, not a physical-memory measurement or proof of paging.
- Both H.264 videos verified at 70 frames, 24 FPS, 2.916667 s. The independent
  verifier checked raw values, source order, exact official color formula,
  code/input/checkpoint hashes and every encoded frame's content/size/FPS.
- Output: outputs/video/rollercoaster_70_518/. Selected contact sheet and JSON
  records are in results/{figures,tables}/video/ with the same run prefix.
- Full available-clip inference is complete. Three-column same-frame quality
  comparison, NYU metric evaluation, and course clarification of 70 vs 100
  frames remain pending.

## 2026-11-11: VDA-Small color compatibility and 32-frame GPU smoke tests

- Installed einops 0.8.2, easydict 1.14, tqdm 4.70.1, imageio 2.38.1,
  imageio-ffmpeg 0.8.1 (plus tqdm's colorama dependency). pip check passed;
  existing torch 2.7.1+cu126 / torchvision 0.22.1+cu126 were preserved.
- Added configs/video.yaml and src/video/run_video.py using the official
  Video Depth Anything-Small relative model, strict checkpoint load, offline
  inference and default FP16 autocast; no external tracked code was modified.
- Official commit: 4f5ae23172ba60fd7bc11ef671cca678842c7072.
  Checkpoint SHA256: 13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609.
- Adapted visualization via matplotlib.colormaps['inferno'].colors; retained
  official 255-index lookup. All pixels of both runs match the original color
  formula exactly; a constant-map guard is included.
- Both runs processed the real first 32 frames at 24 FPS (1.333333 s).
  Official max_res 640 reads 640x360 RGB, returned depth is float32 (32,360,640).
  Model windows/overlap alignment were preserved; context-only padding does not
  extend the returned clip.
- smoke_32_518 passed: cold pipeline 33.5644 s, PyTorch peak allocated
  8014.219 MiB / reserved 11858 MiB. Standard forward hooks were bypassed
  by the official direct forward call; no window shapes/timings were captured.
  This limitation is retained in the historical record; the source was snapshotted.
- Corrected the observer to wrap forward without changing tensor computations.
  smoke_32_364 passed: two windows of [1,32,3,364,644], cold pipeline 2.2324 s,
  peak allocated 2385.355 MiB / reserved 3162 MiB. This is the new default
  smoke setting, with substantially more memory headroom. No quality superiority
  or steady-state speed claim is inferred from these short tests.
- scripts/verify_video_run.py independently verified source RGB order, all
  7,372,800 raw values per run, normalization, exact official color formula,
  code/input/checkpoint hashes, and every encoded frame's content/count/FPS.
- Outputs: outputs/video/{smoke_32_518,smoke_32_364}/. Selected contact sheets
  and JSON records: results/{figures,tables}/video/. Both MP4s are H.264,
  written with imageio's bundled FFmpeg 9.0.2.
- Full 70-frame VDA inference and same-frame comparison with DA V2 remain
  pending, as does course clarification of the 70-vs-100-frame discrepancy.

## 2026-11-11: Complete available 70-frame frame-wise clip

- Executed the existing script: python -m src.monocular.run_framewise
  --run-name rollercoaster_70 --max-frames 70. Model/preprocessing unchanged.
- Same Small/vits weights and input hashes as the first-frame validation,
  CUDA FP32, original 960x540, network 924x518, indices 0-69 at original 24 FPS.
- Saved 70 raw float32 maps and 70 lossless RGB/color/comparison images.
  All 36,288,000 prediction values are finite. Frame-0 raw depth matches the
  earlier official-interface-verified run exactly.
- Visualization uses one range [0, 3.818237782] for the whole clip, inferno palette.
- Mean forward plus output resize 54.858271 ms/frame; mean pipeline with
  preprocessing/transfers 79.034686 ms/frame. Excludes decoding, writes and warm-up.
- PyTorch peak allocated 413.570313 MiB, reserved 574 MiB, including weights.
- Original mp4v comparison video: 1932x588, 70 frames, 24 FPS, 2.916667 s.
- Extra compatible H.264/yuv420p video encoded directly from PNGs with local
  FFmpeg 8.1.2 / libopenh264; exact command/version/hashes in h264_encoding.json.
- All saved RGBs match source frames in order; all color/combined PNGs reproduce
  exactly from fixed normalization. Both MP4s fully decoded and checked against
  per-frame PNGs, with correct frame count/FPS and no severe encoding mismatch.
- Outputs: outputs/framewise/rollercoaster_70/. Selected contact sheet and
  verification/provenance: results/{figures,tables}/framewise/rollercoaster_70*.
- User confirmed only the PDF-linked repository video is available. This is a
  real 70-frame run, not a claimed 100-frame completion; course acceptance of
  the source-length discrepancy remains to be clarified. Video-model comparison,
  NYU metric evaluation and Part 3 are still pending.

## 2026-11-11: Frame-wise first-frame validation

- Implemented independent Depth Anything V2-Small inference, default one frame.
- Official commit: a561b849ebae10a6f5ef49e26c83cbbcd36c71bf; tracked code
  unmodified. Strictly loaded Small/vits relative weights.
- GPU: RTX 4070 Laptop; torch 2.7.1+cu126 / torchvision 0.22.1+cu126, FP32.
- Frame 0, 0.000 s, original 960x540 at 24 FPS; network 924x518; output float32
  540x960, all finite, range [0, 3.042626858]. No metric scale fitting.
- Input tensor matches official image2tensor exactly; saved output matches
  official infer_image, maximum absolute difference 0.
- One warm-up, one measured frame: 66.0723 ms forward/resize, 90.9350 ms with
  transforms and transfers. Not a stable performance benchmark. PyTorch peak
  allocated 413.5703 MiB, peak reserved 574 MiB.
- Outputs: outputs/framewise/first_frame/; selected preview/records under
  results/{figures,tables}/framewise/. Twelve unit tests pass.
- Source issue: the local video and repository copy are identical and contain
  only 70 decoded frames. The 100-frame course run remains pending; frames
  must not be repeated, padded or resampled to conceal this discrepancy.
- Video Depth Anything, NYU metric inference and Part 3 have not been run.

## 2026-11-08: Explicit left-right consistency diagnostic

- Added separate signed right-reference matching using the same source inputs
  and baseline parameters, with mirrored minDisparity. Baseline predictions
  remain intact and are checked with SHA256 before/after each scene.
- Inclusive 1 px threshold; nearest-pixel sampling; five distinct states for
  left-invalid / out-of-bounds / right-invalid / inconsistent / consistent.
- MAE 2.033342 -> 1.513458 px, coverage 75.864931% -> 63.719293% (scene means).
- Mean pass fraction over checkable pixels 96.681630%; this is not accuracy.
- Eight unit tests pass, including negative right disparity, no border wrapping,
  subpixel nearest sampling, zero disparity, invalid pixels and threshold edges.
- Full outputs in outputs/stereo/left_right/; selected figures/tables in results/.
- This additional diagnostic does not replace Part 3 or modify the baseline.

## 2026-11-06: Classical stereo completed

- Fixed StereoSGBM baseline, all 15 Middlebury trainingQ scenes; no training.
- CPU: Intel Core i9-14900HX, one OpenCV thread; no GPU use.
- Python 3.11.18; OpenCV package 5.0.0.93; NumPy 2.4.6; Matplotlib 3.11.2;
  PyYAML 6.0.3. Exact direct dependencies: requirements-stereo.txt.
- Baseline blockSize=5; all other settings: configs/stereo.yaml. Search range
  from calibration ndisp, rounded up to 16. No GT-derived vmin/vmax tuning.
- MAE 2.033342 px, coverage 75.864931%, equal scene weighting.
- Pixel-weighted MAE 1.906025 px; coverage 76.339301%.
- Mean matching-only time 38.252120 ms/pair (one warm-up, median of three calls
  per scene, then equal scene averaging). Short CPU timings are load-sensitive.
- Sensitivity: block 3: MAE 1.866076 px / coverage 75.105833%; block 9:
  MAE 2.388422 px / coverage 76.757208%. P1/P2 change with the block formula;
  not a one-factor block-only test. Original block-5 baseline remains unchanged.
- Mandatory evaluation uses all finite positive GT, not mask0nocc.png.
- PFM loader was corrected during preparation to preserve pixel payload values
  instead of multiplying the header magnitude; preliminary invalid metrics were
  discarded. Final PFM arrays match torchvision's reader for all 15 scenes.
- Five unit tests passed; baseline metrics independently recomputed from saved
  predictions. Full run manifests contain exact input and source SHA256 values.
- Raw outputs: outputs/stereo/{baseline,window3,window9}/. Selected results:
  results/{tables,figures}/stereo/. Report: report/stereo/.
- No Part 1 neural frame-wise experiment or Part 2/3 experiment has run yet.

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
