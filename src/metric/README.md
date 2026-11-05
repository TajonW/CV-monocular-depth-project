# Part 2: Metric depth

Implement RGB-only inference for one model (UniDepthV2 ViT-S is the proposed
starting point) and `evaluate_nyu.py` here. Download splits.mat before evaluation.
Evaluate raw metric output on all 654 NYU test images using the course crop,
depth range, and equal per-image averaging. Never align the baseline to test
ground truth. See `docs/evaluation_protocol.md`.
