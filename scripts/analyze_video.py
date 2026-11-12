"""Reproduce qualitative evidence from saved DA/VDA predictions; no inference."""

import csv
import json
import importlib.metadata as metadata
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video.visualization import colorize_relative
from src.monocular.run_framewise import sha256, write_json

OUTPUT = ROOT / "outputs/analysis/video_qualitative"


def load_arrays():
    da_root = ROOT / "outputs/framewise/rollercoaster_70"
    vda_root = ROOT / "outputs/video/rollercoaster_70_518"
    rgb = np.load(vda_root / "input_rgb.npz")["frames"]
    raw_vda = np.load(vda_root / "depths.npz")["depths"]
    raw_da = np.stack([cv2.resize(np.load(da_root / "frames" / f"{i:06d}" / "relative_depth.npy"),
                                 (640, 360), interpolation=cv2.INTER_LINEAR) for i in range(70)])
    norms = [json.loads((root / "normalization.json").read_text()) for root in (da_root, vda_root)]
    colors = [np.stack([colorize_relative(frame, norm["minimum"], norm["maximum"]) for frame in raw])
              for raw, norm in zip((raw_da, raw_vda), norms)]
    normalized = [(raw - norm["minimum"]) / (norm["maximum"] - norm["minimum"])
                  for raw, norm in zip((raw_da, raw_vda), norms)]
    return rgb, colors, normalized


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(path)


def strip(images, indices, path, box=None, panel_size=(320, 180)):
    width, height = panel_size
    gap, header, row_label = 8, 36, 24
    canvas = np.full((header + len(indices) * (height + row_label), width * 3 + gap * 2, 3), 245, np.uint8)
    for column, title in enumerate(("RGB", "DA V2-Small", "VDA-Small")):
        cv2.putText(canvas, title, (column * (width + gap) + 8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (30, 30, 30), 1, cv2.LINE_AA)
    for row, index in enumerate(indices):
        y = header + row * (height + row_label)
        for column, sequence in enumerate(images):
            image = sequence[index]
            if box:
                x1, y1, x2, y2 = box
                image = image[y1:y2, x1:x2]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
            x = column * (width + gap)
            canvas[y + row_label:y + row_label + height, x:x + width] = image
            cv2.putText(canvas, f"frame {index:02d} | {index / 24:.3f} s", (x + 8, y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def track_tree(rgb, normalized):
    """RGB-only diagnostic, not a GT temporal metric or model improvement."""
    start, end = 64, 69
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    mask = np.zeros(gray[start].shape, np.uint8)
    mask[315:353, 555:632] = 255
    seeds = cv2.goodFeaturesToTrack(gray[start], maxCorners=30, qualityLevel=.03,
                                    minDistance=10, mask=mask, blockSize=5)
    if seeds is None:
        raise RuntimeError("No tree RGB feature candidates")
    points = seeds.copy()
    trajectories = [points[:, 0].copy()]
    valid = np.ones(len(seeds), bool)
    fb_max = np.zeros(len(seeds))
    for index in range(start, end):
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(gray[index], gray[index + 1], points, None, **options)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray[index + 1], gray[index], next_points, None, **options)
        fb = np.linalg.norm(back[:, 0] - points[:, 0], axis=1)
        fb_max = np.maximum(fb_max, fb)
        valid &= (status[:, 0] == 1) & (back_status[:, 0] == 1) & (fb < 1.) & (errors[:, 0] < 20.)
        valid &= (next_points[:, 0, 0] > 3) & (next_points[:, 0, 0] < 637)
        valid &= (next_points[:, 0, 1] > 3) & (next_points[:, 0, 1] < 357)
        points = next_points
        trajectories.append(points[:, 0].copy())
    valid_ids = np.flatnonzero(valid)
    if not len(valid_ids):
        raise RuntimeError("No reliable tree RGB feature track")
    # Feature candidates are ordered by RGB corner strength, not depth behavior.
    primary = int(valid_ids[0])
    rows = []
    for feature in valid_ids:
        for order, index in enumerate(range(start, end + 1)):
            x, y = map(float, trajectories[order][feature])
            values = [float(cv2.getRectSubPix(raw[index], (1, 1), (x, y))[0, 0]) for raw in normalized]
            rows.append(dict(feature_id=int(feature), frame_index=index, x=x, y=y,
                             da2_normalized=values[0], vda_normalized=values[1],
                             max_forward_backward_px=float(fb_max[feature]), primary=bool(feature == primary)))
    with (OUTPUT / "tree_rgb_tracks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row["primary"]]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1.8]))
    for axis, index, row in zip(axes[:2], (start, end), (selected[0], selected[-1])):
        axis.imshow(rgb[index][270:360, 520:640])
        axis.plot(row["x"] - 520, row["y"] - 270, "+", color="yellow", markersize=14, markeredgewidth=2)
        axis.set_title(f"Same RGB feature | frame {index}")
        axis.axis("off")
    for key, name in (("da2_normalized", "DA V2-Small"), ("vda_normalized", "VDA-Small")):
        axes[2].plot([row["frame_index"] for row in selected], [row[key] for row in selected], "o-", label=name)
    axes[2].set(xlabel="Original frame index", ylabel="Output / method's fixed full-clip range",
                title="Tracked-tree display-value diagnostic")
    axes[2].grid(alpha=.3)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "tree_track_diagnostic.png", dpi=140)
    plt.close(figure)
    summary = dict(frames=[start, end], seeds=len(seeds), valid_rgb_tracks=len(valid_ids),
                   primary_feature_id=primary, primary_selection="First surviving RGB corner-strength candidate; no depth-based selection",
                   forward_backward_limit_px=1., lk_patch_error_limit=20., normalized_values=selected,
                   interpretation="A tracked display-value example, not metric depth accuracy or a global temporal-error score")
    ranges = []
    for feature in valid_ids:
        track = [row for row in rows if row["feature_id"] == int(feature)]
        ranges.append(dict(feature_id=int(feature),
                           da2_range=float(np.ptp([row["da2_normalized"] for row in track])),
                           vda_range=float(np.ptp([row["vda_normalized"] for row in track]))))
    summary["all_track_ranges"] = ranges
    summary["vda_smaller_display_range_count"] = sum(row["vda_range"] < row["da2_range"] for row in ranges)
    summary["limitation"] = "One spatially clustered ROI and six frames, arbitrary method-specific relative scales; no independent-scene sample or GT depth/pose. True depth changes cannot be separated from prediction variation."
    (OUTPUT / "tree_track_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def save_review_video():
    destination = OUTPUT / "comparison_slow_4x.mp4"
    with imageio.get_writer(str(destination), fps=6, macro_block_size=1, codec="libx264",
                            ffmpeg_params=["-crf", "18"]) as writer:
        for index in range(70):
            source = ROOT / "outputs/comparison/rollercoaster_70/frames" / f"{index:06d}.png"
            rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
            panel = cv2.copyMakeBorder(rgb, 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
            cv2.putText(panel, "REVIEW COPY: 4x slower, 6 FPS (source 24 FPS); timestamps above remain source time.",
                        (12, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .58, (30, 30, 30), 1, cv2.LINE_AA)
            writer.append_data(panel)
    capture = cv2.VideoCapture(str(destination))
    count = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    while capture.read()[0]:
        count += 1
    capture.release()
    if count != 70 or not np.isclose(fps, 6):
        raise RuntimeError("Slow review frame count/FPS mismatch")
    return dict(path=destination.name, frames=count, fps=fps, duration_s=count / fps,
                sha256=sha256(destination), playback_only=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rgb, colors, normalized = load_arrays()
    images = [rgb, *colors]
    for name, indices in (("overview_early", [0, 4, 8, 12, 16, 20]),
                          ("overview_middle", [24, 28, 32, 36, 40, 44]),
                          ("overview_late", [48, 52, 56, 60, 64, 69])):
        strip(images, indices, OUTPUT / f"{name}.png")
    cases = [
        dict(name="vehicle_34_39", frames=list(range(34, 40)), box=[120, 85, 460, 245], display=[340, 160]),
        dict(name="fine_support_00_05", frames=list(range(0, 6)), box=[330, 15, 600, 175], display=[324, 192]),
        dict(name="junction_54_59", frames=list(range(54, 60)), box=[180, 160, 430, 350], display=[300, 228]),
        dict(name="tree_64_69", frames=list(range(64, 70)), box=[460, 250, 640, 360], display=[360, 220]),
    ]
    for case in cases:
        strip(images, case["frames"], OUTPUT / f"{case['name']}.png", case["box"], tuple(case["display"]))
    (OUTPUT / "case_definitions.json").write_text(json.dumps(cases, indent=2))
    track_tree(rgb, normalized)
    review = save_review_video()
    source_manifest = ROOT / "outputs/comparison/rollercoaster_70/manifest.json"
    frozen = json.loads(source_manifest.read_text())
    for relative, digest in frozen["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Original input changed: {relative}")
    manifest = dict(source_comparison_manifest_sha256=sha256(source_manifest),
                    all_original_input_hashes_still_match=True,
                    script_sha256=sha256(Path(__file__)),
                    helper_sha256={name: sha256(ROOT / name) for name in ("src/video/visualization.py", "src/monocular/run_framewise.py")},
                    cases=cases, review_video=review,
                    packages={name: metadata.version(name) for name in ("numpy", "opencv-python", "matplotlib", "imageio", "imageio-ffmpeg")},
                    inference_performed=False, prediction_arrays_modified=False,
                    normalization="Unchanged full-clip per-method ranges; no per-frame adjustment",
                    temporal_diagnostic="RGB-only pyramidal Lucas-Kanade tracking; not GT temporal error")
    write_json(OUTPUT / "analysis_manifest.json", manifest)
    shutil.copy2(Path(__file__), OUTPUT / "analyze_video_snapshot.py")
    figures = ROOT / "results/figures/video_analysis"
    tables = ROOT / "results/tables/video_analysis"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.glob("*.png"):
        shutil.copy2(path, figures / path.name)
    for name in ("case_definitions.json", "tree_track_diagnostic.json", "tree_rgb_tracks.csv", "analysis_manifest.json"):
        shutil.copy2(OUTPUT / name, tables / name)
    print("Saved overview strips:", OUTPUT)
    # Screening only: fixed-screen differences include scene/camera motion.
    # They must not be interpreted as temporal error or a quality ranking.
    for name, raw in zip(("DA", "VDA"), normalized):
        jumps = np.abs(np.diff(raw, axis=0)).mean(axis=(1, 2))
        top = np.argsort(jumps)[-8:][::-1]
        print(name, "screening transitions (not temporal error):", [(int(i), int(i+1), float(jumps[i])) for i in top])


if __name__ == "__main__":
    main()
