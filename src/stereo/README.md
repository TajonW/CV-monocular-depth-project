# Part 1: Stereo geometry

Implement `run_stereo.py` and `evaluate_stereo.py` here.
Start with Adirondack, then process all 15 trainingQ scenes.

- Read im0.png, im1.png, calib.txt, and disp0GT.pfm.
- Compute OpenCV StereoSGBM disparity and divide the fixed-point result by 16.
- Use quarter-resolution calibration without scaling it again.
- Convert to meters: Z = f * baseline / (disparity + doffs) / 1000.
- Keep an explicit invalid prediction mask and floating-point outputs.
- Report disparity MAE, coverage, and runtime.

See `docs/evaluation_protocol.md` for mask and aggregation rules.
