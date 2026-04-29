from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8_bgr(frame: np.ndarray, name: str = "frame") -> np.ndarray:
    if frame is None:
        raise ValueError(f"{name} is None.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3, got {frame.shape}.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def resize_if_needed(frame: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    if width is None and height is None:
        return frame
    h, w = frame.shape[:2]
    if width is None:
        width = int(round(w * (height / h)))
    if height is None:
        height = int(round(h * (width / w)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def bgr_to_tensor(frame: np.ndarray, device):
    import torch

    frame = ensure_uint8_bgr(frame)
    arr = np.ascontiguousarray(frame.transpose(2, 0, 1))
    tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    return tensor.unsqueeze(0) / 255.0


def tensor_to_bgr(tensor) -> np.ndarray:
    tensor = tensor.detach().float().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).round()
    return np.clip(arr, 0, 255).astype(np.uint8)


def read_image(path: str) -> np.ndarray:
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return frame


def write_image(path: str, frame: np.ndarray) -> None:
    ok = cv2.imwrite(path, ensure_uint8_bgr(frame))
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")
