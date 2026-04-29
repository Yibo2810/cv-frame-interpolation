from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from vfi.backends.base import InterpolationBackend
from vfi.device import resolve_device
from vfi.frame import ensure_uint8_bgr, tensor_to_bgr


class RIFEExternalBackend(InterpolationBackend):
    """Adapter around the official ECCV2022-RIFE repository.

    This class does not vendor RIFE code. It imports a local checkout and loads
    pretrained weights from its train_log directory.
    """

    name = "rife"

    def __init__(
        self,
        repo_path: str = "external/ECCV2022-RIFE",
        model_dir: str | None = None,
        device: str = "auto",
        scale: float = 1.0,
        tta: bool = False,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else self.repo_path / "train_log"
        self.device = resolve_device(device)
        self.scale = float(scale)
        self.tta = bool(tta)
        self.model = self._load_model()

    def _load_model(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"RIFE repo not found at {self.repo_path}. "
                "Clone ECCV2022-RIFE there or pass --rife-root."
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"RIFE model directory not found at {self.model_dir}. "
                "Download pretrained weights into train_log or pass --rife-model-dir."
            )

        sys.path.insert(0, str(self.repo_path))
        try:
            model_cls, variant = self._import_model_class()
            self._patch_external_device()
            model = model_cls()
            model.load_model(str(self.model_dir), -1)
            model.eval()
            if hasattr(model, "flownet"):
                model.flownet.to(self.device)
            print(f"Loaded RIFE variant: {variant} on {self.device}")
            return model
        finally:
            try:
                sys.path.remove(str(self.repo_path))
            except ValueError:
                pass

    @staticmethod
    def _import_model_class():
        candidates = [
            ("model.RIFE_HDv2", "RIFE_HDv2"),
            ("train_log.RIFE_HDv3", "RIFE_HDv3"),
            ("model.RIFE_HD", "RIFE_HD"),
            ("model.RIFE", "RIFE"),
        ]
        last_error: Exception | None = None
        for module_name, variant in candidates:
            try:
                module = importlib.import_module(module_name)
                return module.Model, variant
            except Exception as exc:  # external repo variants differ by weight package
                last_error = exc
        raise ImportError("Could not import a RIFE Model class from the external repo.") from last_error

    def _patch_external_device(self) -> None:
        module_names = [
            "model.RIFE_HDv2",
            "train_log.RIFE_HDv3",
            "model.RIFE_HD",
            "model.RIFE",
            "model.warplayer",
            "train_log.IFNet_HDv3",
            "model.IFNet",
            "model.IFNet_m",
        ]
        for module_name in module_names:
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "device"):
                setattr(module, "device", self.device)
            if module_name == "model.warplayer" and module is not None and hasattr(module, "backwarp_tenGrid"):
                module.backwarp_tenGrid.clear()

    @staticmethod
    def _bgr_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
        arr = np.ascontiguousarray(frame.transpose(2, 0, 1))
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0

    def _run_inference(self, img0: torch.Tensor, img1: torch.Tensor, t: float) -> torch.Tensor:
        signature = inspect.signature(self.model.inference)
        params = signature.parameters
        kwargs = {}
        if "scale" in params:
            kwargs["scale"] = self.scale
        if "TTA" in params:
            kwargs["TTA"] = self.tta
        if "timestep" in params:
            kwargs["timestep"] = t
        elif abs(t - 0.5) > 1e-6:
            raise RuntimeError("This RIFE variant only supports midpoint inference through this adapter.")
        return self.model.inference(img0, img1, **kwargs)

    @torch.inference_mode()
    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        frame0 = ensure_uint8_bgr(frame0, "frame0")
        frame1 = ensure_uint8_bgr(frame1, "frame1")
        if frame0.shape != frame1.shape:
            frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]), interpolation=cv2.INTER_AREA)

        img0 = self._bgr_to_tensor(frame0, self.device)
        img1 = self._bgr_to_tensor(frame1, self.device)
        _, _, height, width = img0.shape
        padded_h = ((height - 1) // 32 + 1) * 32
        padded_w = ((width - 1) // 32 + 1) * 32
        padding = (0, padded_w - width, 0, padded_h - height)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        pred = self._run_inference(img0, img1, t=t)
        pred = pred[:, :, :height, :width]
        return tensor_to_bgr(pred)
