# Source code

Stereo, frame-wise depth, offline video depth, three-column comparison and
NYU metric-model evaluation are implemented and verified. Improvement and
application folders contain implementation notes. Keep absolute machine-specific
paths out of source files; read paths from configurations or CLI arguments.

- `stereo`: Middlebury stereo matching, metric conversion, and evaluation.
- `monocular`: independent Depth Anything V2-Small frame predictions.
- `metric`: one RGB-only metric model and NYU evaluation.
- `video`: Video Depth Anything-Small inference and comparisons.
- `improvement`: the single improvement selected after baseline verification.
- `application`: a simple depth-aware effect.
- `utils`: shared loaders, prediction storage, and visualization.
