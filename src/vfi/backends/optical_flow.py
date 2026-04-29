from __future__ import annotations

import cv2
import numpy as np

from vfi.backends.base import InterpolationBackend
from vfi.frame import ensure_uint8_bgr


_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}


def _make_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return x, y


def _backward_warp(frame: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    height, width = frame.shape[:2]
    grid_x, grid_y = _make_grid(width, height)
    map_x = grid_x - scale * flow[..., 0].astype(np.float32)
    map_y = grid_y - scale * flow[..., 1].astype(np.float32)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class DISFlowBackend(InterpolationBackend):
    name = "flow"

    def __init__(
        self,
        preset: str = "fast",
        photometric_threshold: float = 35.0,
        use_farneback_fallback: bool = True,
    ) -> None:
        preset = preset.lower()
        if preset not in _PRESETS:
            raise ValueError(f"Unknown DIS preset {preset!r}; choose one of {sorted(_PRESETS)}.")
        self.preset = preset
        self.photometric_threshold = float(photometric_threshold)
        self.use_farneback_fallback = use_farneback_fallback
        self.flow = cv2.DISOpticalFlow_create(_PRESETS[preset])

    def _calc_flow(self, gray0: np.ndarray, gray1: np.ndarray) -> np.ndarray:
        try:
            return self.flow.calc(gray0, gray1, None)
        except cv2.error:
            if not self.use_farneback_fallback:
                raise
            return cv2.calcOpticalFlowFarneback(
                gray0,
                gray1,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        frame0 = ensure_uint8_bgr(frame0, "frame0")
        frame1 = ensure_uint8_bgr(frame1, "frame1")
        if frame0.shape != frame1.shape:
            frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]), interpolation=cv2.INTER_AREA)

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow01 = self._calc_flow(gray0, gray1)
        flow10 = self._calc_flow(gray1, gray0)

        warp0 = _backward_warp(frame0, flow01, t)
        warp1 = _backward_warp(frame1, flow10, 1.0 - t)
        warped_mid = cv2.addWeighted(warp0, 1.0 - t, warp1, t, 0.0)

        direct_blend = cv2.addWeighted(frame0, 1.0 - t, frame1, t, 0.0)
        disagreement = np.mean(
            np.abs(warp0.astype(np.float32) - warp1.astype(np.float32)),
            axis=2,
            keepdims=True,
        )
        confidence = 1.0 - np.clip(disagreement / self.photometric_threshold, 0.0, 1.0)
        out = confidence * warped_mid.astype(np.float32) + (1.0 - confidence) * direct_blend.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

