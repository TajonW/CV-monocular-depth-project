# Source code

Implement project code in the module folders below. No inference or evaluation
runner has been implemented yet. Keep absolute machine-specific paths out of
source files; read paths from configurations or command-line arguments.

- `stereo`: Middlebury stereo matching, metric conversion, and evaluation.
- `monocular`: independent Depth Anything V2-Small frame predictions.
- `metric`: one RGB-only metric model and NYU evaluation.
- `video`: Video Depth Anything-Small inference and comparisons.
- `improvement`: the single improvement selected after baseline verification.
- `application`: a simple depth-aware effect.
- `utils`: shared loaders, prediction storage, and visualization.
