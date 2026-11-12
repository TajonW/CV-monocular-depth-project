# Frame-wise relative depth

The entry point `python -m src.monocular.run_framewise` processes each video
frame independently with the official Depth Anything V2-Small (vits) model.
Run all commands from the repository root.

## Reproduce the Windows CPU run

Run the following commands in Anaconda Prompt with the existing Python 3.10 `cvproject3` environment.

```bat
conda activate cvproject3
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-framewise.txt
python scripts/prepare_framewise.py
python -m src.monocular.run_framewise --inspect-only --frames 70
python -m src.monocular.run_framewise --frames 5 --device cpu --cpu-threads 2 --output-dir outputs/framewise/dav2_small_pilot5
python -m src.monocular.run_framewise --frames 70 --device cpu --cpu-threads 2 --output-dir outputs/framewise/dav2_small_70
```

Choose a fresh output directory when repeating a run. Existing run directories
are protected against overwriting. The first requested frames are decoded and
checked before model initialization. The CLI defaults to 100 frames and rejects
a shorter input instead of silently extending it.

## Supplied clip length

The clip linked by the assignment decodes to **70 frames**, indexed **0..69**,
at **24 fps**, with native size **960 x 540**. The user approved processing all
70 frames and documenting the difference from the requested 100-frame prefix.
The pinned source URL and SHA-256 are in `scripts/prepare_framewise.py`.
The optional 195-frame clip is outside the Part 1 evaluation scope. For
Part 2, compare the same source file and frame indices 0..69 at 24 fps.

## NVIDIA GPU or Colab commands

These GPU commands are prepared for handoff and have **not been executed**.
Transfer this working tree, including this branch's new files, to the GPU host
or upload it into Colab, then use its repository root as the working directory.
For Colab, select a GPU runtime and run the shell commands in a `%%bash` cell.
Use a Python environment supported by PyTorch 2.6 and a CUDA 12.4-compatible
NVIDIA driver.

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-framewise.txt
python scripts/prepare_framewise.py
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m src.monocular.run_framewise --inspect-only --frames 70
python -m src.monocular.run_framewise --frames 5 --device cuda --output-dir outputs/framewise/dav2_small_gpu_pilot5
python -m src.monocular.run_framewise --frames 70 --device cuda --output-dir outputs/framewise/dav2_small_gpu_70
```

The GPU path uses float32 and synchronized timing. Its actual speed and peak
GPU memory will be recorded by the run. The checkpoint and model architecture
follow the [official model usage](https://github.com/DepthAnything/Depth-Anything-V2).
The installation commands follow [PyTorch's version matrix](https://pytorch.org/get-started/previous-versions/).

## Outputs and interpretation

The reference CPU run is stored in `outputs/framewise/dav2_small_70/`.
Each run records predictions, video and metadata in its output directory.

- `predictions/depth_000000.npy`, etc. are the original float32 relative-depth
  predictions resized to source resolution using the official interpolation.
- `frame_indices.npy` and `frames.csv` retain zero-based source frame indices,
  timestamps (`index / source_fps`), source-frame pixel hashes, dimensions,
  tensor shape, value ranges, validity and inference time.
- `rgb_depth.mp4` uses the source frame rate and side-by-side RGB/depth.
  The depth uses OpenCV TURBO with one global min/max from the run's raw
  predictions. The pilot and full run each have their own recorded range.
- `preview_*.png` shows the first, middle and last selected frames using that
  same run-wide range.
- `run.json` records source/checkpoint hashes, upstream code commit and status,
  device, versions, native and tensor dimensions, warm-up and inference timing,
  and the visualization range. Its presence with `completed=true` marks a
  successfully exported run.
- The reference run also has `acceptance.json` with its saved-output checks
  and timing summary.

Timing includes preprocessing, transfer, forward inference, native-resolution
interpolation and transfer back to CPU. Decoding, output writes and rendering
are outside the inference timer. One warm-up inference is recorded separately.
The Windows run uses two PyTorch CPU threads and one OpenCV thread.

All predictions are relative depth. Retain the full run's visualization range
when visualizing any future improvement of this method. The intended comparison
is qualitative under the course protocol.
