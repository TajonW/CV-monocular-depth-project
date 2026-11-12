# Selected results

## Middlebury stereo

- [Middlebury stereo metrics](tables/middlebury_stereo.csv) lists the 15 scenes
  and their equal-weight scene mean.
- [Adirondack](figures/middlebury_adirondack.png) shows the reproduction scene.
- [PianoL](figures/middlebury_pianol.png) shows the scene with the highest MAE.

The figures show left RGB, estimated and reference disparity on a shared
scale within each scene, invalid predictions, absolute error on the valid
overlap, and metric depth. Complete outputs are under the ignored
`outputs/stereo/` directory.

## Frame-wise relative depth

[Selected rollercoaster frames](figures/framewise_dav2_small_rollercoaster.jpg)
show decoded frames 0, 35 and 69 from the RGB/depth video. Depth Anything
V2-Small used one OpenCV TURBO range of `[0, 3.818236]` across all 70
source frames at 24 fps. The colors represent relative depth.

CPU inference averaged 4.742 s/frame with a median of 3.009 s/frame, including
the recorded 114.008 s timing at frame 28. The full MP4, raw float32 predictions
and run metadata are under `outputs/framewise/dav2_small_70/`.
See the [experiment log](../docs/experiment_log.md) for timing and validation.

## NYU metric depth

UniDepthV2 ViT-S used RGB-only CPU inference on all 654 official NYU Depth V2
test images. The table reports equal-weight means across images.

| Images | AbsRel | RMSE (m) | δ1 |
| ---: | ---: | ---: | ---: |
| 654 | 0.114609 | 0.398949 | 89.2103% |

The evaluation uses finite ground truth in `[0.001, 10]` m within the course
crop. Full predictions and per-image scores are under
`outputs/nyu/unidepth_v2_vits14_test654/`. See the
[experiment log](../docs/experiment_log.md) for protocol and verification.
