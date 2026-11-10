# Source code

The stereo baseline is implemented in `stereo/run_stereo.py`. The other
directories describe planned modules. Read machine-specific paths from
configuration or command-line arguments.

- `stereo`: Middlebury stereo matching, metric conversion, and evaluation.
- `monocular`: independent Depth Anything V2-Small frame predictions.
- `metric`: one RGB-only metric model and NYU evaluation.
- `video`: Video Depth Anything-Small inference and comparisons.
- `improvement`: the single improvement selected after baseline verification.
- `application`: a simple depth-aware effect.
- `utils`: shared loaders, prediction storage, and visualization.
