# Experiment configurations

`stereo.yaml` records the implemented fixed StereoSGBM baseline. The runner
loads it explicitly and stores the effective configuration and per-scene OpenCV
parameters with every run.

`framewise.yaml` configures Depth Anything V2-Small, with a safe default of one
frame for validation. Native output resolution and official preprocessing are
preserved. `--max-frames` requests larger prefixes; the video must actually
contain that many frames. Effective settings and a clip-wide visualization
range are saved per run.

`video.yaml` configures the official Video Depth Anything-Small offline FP16
GPU runner. The safe smoke-test prefix is 32 frames, max_res 640, input_size
364 (verified with memory headroom). Input_size 518 also completed but used
near-8-GiB allocation. The official 32-frame temporal window settings are kept.

`metric.yaml` records the NYU data/split, pinned UniDepth source/Small weights,
two warm-ups and thread setting. The runner preserves official infer(rgb)
preprocessing/FP16 autocast, native meter output, and the course crop/range.
No test-scale fitting or GT camera input is used.
