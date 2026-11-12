"""Unmodified official UniDepthV2.infer(rgb), local strict Small checkpoint."""

import inspect
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


class MetricDepth:
    def __init__(self, repository, checkpoint):
        runtime = ROOT / "tmp/unidepth_runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(runtime)
        os.environ.update(WANDB_MODE="disabled", WANDB_DISABLED="true", HF_HUB_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TORCH_HOME=str(runtime / "torch"))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this configured UniDepthV2 run")
        sys.path.insert(0, str(Path(repository).resolve()))
        from unidepth.models import UniDepthV2
        if not Path(inspect.getfile(UniDepthV2)).resolve().is_relative_to(Path(repository).resolve()):
            raise ImportError("Unexpected UniDepth source location")
        self.model = UniDepthV2.from_pretrained(str(checkpoint), local_files_only=True, strict=True)
        # Explicitly verify the local safe state also has no missing/unexpected tensors.
        from safetensors.torch import load_file
        self.model.load_state_dict(load_file(str(Path(checkpoint) / "model.safetensors")), strict=True)
        self.model = self.model.to("cuda").eval()
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.network_shape = None
        self.handle = self.model.pixel_encoder.register_forward_pre_hook(self._observe)

    def _observe(self, module, inputs):
        self.network_shape = list(inputs[0].shape)

    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError("Expected native NYU RGB uint8")
        torch.cuda.synchronize()
        start = perf_counter()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        # infer owns normalization, resize, FP16 CUDA autocast and native-size postprocess.
        output = self.model.infer(tensor)
        depth = output["depth"]
        returned_dtype = str(depth.dtype)
        prediction = depth[0, 0].float().cpu().numpy().copy()
        torch.cuda.synchronize()
        elapsed = (perf_counter() - start) * 1000
        if prediction.shape != (480, 640) or not np.isfinite(prediction).all():
            raise RuntimeError("Invalid raw metric prediction; no silent masking/rescaling")
        return prediction, dict(inference_pipeline_ms=elapsed, network_shape=self.network_shape,
                                returned_dtype=returned_dtype)
