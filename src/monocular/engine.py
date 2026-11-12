"""Depth Anything V2-Small inference using the official model and transforms."""

from contextlib import nullcontext
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


def load_official_code(repository):
    repository = Path(repository).resolve()
    if not (repository / "depth_anything_v2/dpt.py").is_file():
        raise FileNotFoundError(f"Clone the official Depth-Anything-V2 repository at {repository}")
    sys.path.insert(0, str(repository))
    import depth_anything_v2.dpt as official
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    if not Path(official.__file__).resolve().is_relative_to(repository):
        raise ImportError("A different depth_anything_v2 installation was imported")
    return official.DepthAnythingV2, Resize, NormalizeImage, PrepareForNet


class FramewiseDepth:
    """No frame history, scale fitting, temporal smoothing, or metric conversion."""

    def __init__(self, repository, checkpoint, input_size=518, device="auto", precision="fp32"):
        if input_size < 14 or input_size % 14:
            raise ValueError("input_size must be a positive multiple of 14 (default 518)")
        if device not in ("auto", "cuda", "cpu") or precision not in ("fp32", "fp16"):
            raise ValueError("Unsupported device or precision")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
        if device == "cpu" and precision != "fp32":
            raise ValueError("Use fp32 for CPU inference")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Small checkpoint missing: {checkpoint}")
        self.device = torch.device(device)
        self.precision = precision
        self.input_size = input_size
        cls, Resize, NormalizeImage, PrepareForNet = load_official_code(repository)
        # Same BGR->RGB, resize, normalization and CHW transform as image2tensor.
        # Explicit device placement avoids its automatic CUDA choice in --device cpu mode.
        self.transform = Compose([
            Resize(width=input_size, height=input_size, resize_target=False,
                   keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound",
                   image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        self.model = cls(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model = self.model.to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def prepare(self, bgr):
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("Expected OpenCV BGR uint8 HxWx3 input")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
        array = self.transform({"image": rgb})["image"]
        return torch.from_numpy(array).unsqueeze(0).to(self.device)

    def autocast_context(self):
        return (torch.autocast("cuda", dtype=torch.float16)
                if self.precision == "fp16" else nullcontext())

    def predict(self, bgr):
        """Output float32 at original frame size, plus explicitly scoped timings."""
        self.synchronize()
        start = perf_counter()
        tensor = self.prepare(bgr)
        self.synchronize()
        prepared = perf_counter()
        with torch.inference_mode(), self.autocast_context():
            depth = self.model(tensor)
            depth = F.interpolate(depth[:, None], bgr.shape[:2], mode="bilinear", align_corners=True)[0, 0]
        self.synchronize()
        inferred = perf_counter()
        result = depth.float().cpu().numpy().copy()
        transferred = perf_counter()
        if result.shape != bgr.shape[:2] or not np.isfinite(result).all():
            raise RuntimeError("Model returned wrong-size or nonfinite depth; output was not silently masked")
        if np.any(result < 0):
            raise RuntimeError("Unexpected negative output from the relative-depth model")
        return result, dict(
            input_tensor_shape=list(tensor.shape),
            preprocess_ms=(prepared-start)*1000,
            inference_ms=(inferred-prepared)*1000,
            to_cpu_ms=(transferred-inferred)*1000,
            pipeline_ms=(transferred-start)*1000,
        )
