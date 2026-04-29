from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from vfi.backends.base import InterpolationBackend
from vfi.device import resolve_device
from vfi.frame import bgr_to_tensor, ensure_uint8_bgr, tensor_to_bgr


class TinyVFIModule(nn.Module):
    """A tiny PyTorch VFI module used to validate the model/backend path.

    The final layer is zero-initialized, so the untrained model starts as a
    stable linear interpolator. Later, this module can be replaced by RIFE.
    """

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        last_conv = self.net[-2]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        batch, _, height, width = img0.shape
        tmap = torch.full((batch, 1, height, width), float(t), dtype=img0.dtype, device=img0.device)
        linear = (1.0 - t) * img0 + t * img1
        residual = 0.05 * self.net(torch.cat([img0, img1, tmap], dim=1))
        return (linear + residual).clamp(0.0, 1.0)


class TorchLiteBackend(InterpolationBackend):
    name = "torch"

    def __init__(
        self,
        device: str = "auto",
        weights: str | None = None,
        compile_model: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.model = TinyVFIModule().to(self.device).eval()
        if weights:
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state)
        if compile_model:
            self.model = torch.compile(self.model)

    @torch.inference_mode()
    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        frame0 = ensure_uint8_bgr(frame0, "frame0")
        frame1 = ensure_uint8_bgr(frame1, "frame1")
        if frame0.shape != frame1.shape:
            frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]), interpolation=cv2.INTER_AREA)

        img0 = bgr_to_tensor(frame0, self.device)
        img1 = bgr_to_tensor(frame1, self.device)
        pred = self.model(img0, img1, t=t)
        return tensor_to_bgr(pred)

