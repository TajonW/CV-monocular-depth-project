# Part 1: Frame-wise relative depth

Implement `run_framewise.py` here using Depth Anything V2-Small.
Predict the specified clip's first 100 frames independently.
Save floating-point predictions before colorization, preserve frame rate, and
label the output relative depth. Use one fixed normalization over the clip;
the official demo's per-frame normalization must be replaced for submission.
