"""Modern Matplotlib API with the official VDA color-index convention."""

import cv2
import imageio.v2 as imageio
from matplotlib import colormaps
import numpy as np


def colorize_relative(depth, lower, upper, grayscale=False):
    if not np.isfinite(depth).all() or not np.isfinite([lower, upper]).all() or upper < lower:
        raise ValueError("Nonfinite depth or invalid display range")
    # Keep the official 0..255 integer lookup; only the colormap API changes.
    normalized = (np.clip((depth - lower) / (upper - lower), 0, 1)
                  if upper > lower else np.zeros_like(depth))
    indices = (normalized * 255).astype(np.uint8)
    if grayscale:
        return indices
    return (np.asarray(colormaps["inferno"].colors)[indices] * 255).astype(np.uint8)


def save_video(frames, output_video_path, fps=24, is_depths=False, grayscale=False):
    """Compatible replacement for utils.dc_utils.save_video; RGB input/output."""
    if len(frames) == 0 or fps <= 0 or not np.isfinite(fps):
        raise ValueError("Expected nonempty frames and positive finite FPS")
    lower, upper = (float(frames.min()), float(frames.max())) if is_depths else (None, None)
    with imageio.get_writer(str(output_video_path), fps=fps, macro_block_size=1,
                            codec="libx264", ffmpeg_params=["-crf", "18"]) as writer:
        for frame in frames:
            writer.append_data(colorize_relative(frame, lower, upper, grayscale) if is_depths else frame)


def comparison_image(rgb, color_depth, index, fps):
    height, width = rgb.shape[:2]
    panel = np.full((height + 48, width * 2 + 12, 3), 245, np.uint8)
    panel[48:, :width] = rgb
    panel[48:, width + 12:] = color_depth
    cv2.putText(panel, f"RGB | frame {index:06d} | {index / fps:.3f} s", (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(panel, "Video Depth Anything-Small | relative depth", (width + 24, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    return panel
