# Selected results

Track small, selected outputs here: CSV tables in `tables/` and comparison or
failure-case images in `figures/`. Do not commit full prediction arrays, entire
video collections, or dataset copies. Complete outputs belong in the ignored
`outputs/` directory.

Completed stereo tables and run provenance are in `tables/stereo/`. Selected
comparison figures and explicit failure ROIs are in `figures/stereo/`.
Run `python scripts/summarize_stereo.py` after generating the three named runs
to regenerate these selected artifacts.

Depth Anything V2-Small previews for the first-frame and complete available
70-frame runs are in `figures/framewise/`. Configuration/provenance, statistics,
official first-frame verification and all-frame clip checks are in
`tables/framewise/`. Full videos/raw predictions remain in local outputs.

Video Depth Anything-Small 32-frame GPU smoke-test contact sheets are in
`figures/video/`. Both input_size 518 and 364 passed; manifests, scoped cold
timings, memory, normalization and independent output checks are in
`tables/video/`. Full raw arrays and H.264 videos remain in `outputs/video/`.
These checks do not establish full-clip temporal quality or metric accuracy.

The full available 70-frame VDA-Small run at input_size 518 is also verified.
Its selected artifacts use the prefix `rollercoaster_70_518_`; the corresponding
H.264 videos and raw arrays are in `outputs/video/rollercoaster_70_518/`.
The same-frame three-column RGB/DA V2/VDA comparison is generated and verified.
Selected contact sheets are in `figures/comparison/`; manifests and checks are
in `tables/comparison/`. The H.264 video and all lossless comparison frames are
in `outputs/comparison/rollercoaster_70/`. Qualitative evidence and analysis are
documented in `docs/video_qualitative_analysis.md`, with consecutive ROI strips
and an RGB-tracked tree diagnostic in `figures/video_analysis/`. All tracks,
ROI definitions and provenance are in `tables/video_analysis/`.

UniDepthV2 ViT-S completed all 654 NYU test images under the course protocol.
`tables/metric/nyu_test654_metrics.csv` contains per-image results; summary,
manifest, independent metric verification and figure selections share that
prefix. RGB/GT/prediction/error panels are in `figures/metric/`. Representative
examples and three post-evaluation high-error cases are included. Full raw
meter predictions stay in `outputs/metric/nyu_test654/`.
