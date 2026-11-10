# Evaluation protocol

This file summarizes the course protocol. Consult the original assignment in
the local `docs/assignment/` folder for the authoritative requirements.

## Middlebury stereo

- Use the 15 quarter-resolution trainingQ scenes and left-view disp0GT.pfm.
- Keep the native image size and corresponding calibration.
- Convert OpenCV fixed-point disparity to pixels by dividing by 16.
- Ground-truth validity: finite disparity strictly greater than zero.
- Prediction validity: finite disparity with raw OpenCV disparity strictly
  greater than `(minDisparity - 1) * 16`. Include non-sentinel zero disparity
  in the formal run. The legacy reproduction additionally requires `d > 0`.
- Compute MAE on the intersection of valid ground truth and valid predictions.
- Coverage is the number of pixels in that intersection divided by the number
  of valid ground-truth pixels. Report both MAE and coverage.
- Exclude invalid predictions and nonpositive denominators from depth conversion.
- Report per-scene metrics. For the summary, use equal scene averaging and label
  it explicitly. This aggregation choice is a project convention.
- Left-right consistency is optional.

## NYU metric depth

- Use trainNdxs (795 images) for development and testNdxs (654 images) for testing.
- Subtract one from MATLAB indices in Python.
- With h5py: RGB is images[i].transpose(2, 1, 0); filled metric depth is depths[i].T.
- Use filled depths, not rawDepths; keep floating-point values in meters.
- Resize predictions to height 480, width 640.
- Evaluate crop [45:471, 41:601] at valid ground-truth depths between 0.001 and 10 m.
- Clip predictions to [0.001, 10] m. Diagnose nonfinite predictions explicitly;
  do not silently remove hard pixels based on prediction quality.
- Report AbsRel, RMSE, and delta1 (max(pred/gt, gt/pred) < 1.25).
- Compute each metric per image, then average equally over all 654 test images.
- No test-set scale alignment. For a calibration improvement, fit only on
  development images and freeze the resulting global scale before testing.

## Video

- Use the first 100 frames of davis_rollercoaster.mp4 at the original frame rate.
- Compare Depth Anything V2-Small and Video Depth Anything-Small relative depth.
- Use the same palette and a fixed normalization per method over the full clip.
- Reuse the unchanged method's normalization for its own improvement.
- Do not interpret colors as a shared metric scale.
- Required evaluation is qualitative: flicker, edge stability, motion artifacts.
- Video ground-truth accuracy and numerical temporal metrics are not required.

## Efficiency and reproducibility

Record hardware, model/checkpoint, external repository commit, precision, input
and output sizes, inference time per image/frame, and peak GPU memory. State
whether timing includes preprocessing and postprocessing; synchronize CUDA when
timing GPU inference. Separate warm-up from measured inference.

Keep the same images, evaluation masks, output size, and metrics between a
baseline and its improvement. Preserve each model's required preprocessing.
