# Part 1: Stereo geometry

Run `python -m src.stereo.run_stereo` from the repository root to process all
15 quarter-resolution Middlebury trainingQ scenes. Use `--scene Adirondack`
to run only that scene and `--legacy-positive-only` to reproduce the old
positive-only prediction mask.

The runner uses the native calibration, converts OpenCV fixed-point disparity
to pixels, and converts positive denominators to metric depth with
`Z = f * baseline / (disparity + doffs) / 1000`. Full outputs include the raw
fixed-point prediction, masked floating-point disparity and depth, invalid
mask, metrics and overview. The batch CSV includes an equal-weight scene mean.

See [the evaluation protocol](../../docs/evaluation_protocol.md) for masks and
[the project README](../../README.md#part-1-stereo-baseline-cpu) for setup.
