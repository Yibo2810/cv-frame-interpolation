from __future__ import annotations

import cv2
import numpy as np

from vfi.backends.base import InterpolationBackend
from vfi.frame import ensure_uint8_bgr


class LinearBlendBackend(InterpolationBackend):
    name = "blend"

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        frame0 = ensure_uint8_bgr(frame0, "frame0")
        frame1 = ensure_uint8_bgr(frame1, "frame1")
        if frame0.shape != frame1.shape:
            frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]), interpolation=cv2.INTER_AREA)
        return cv2.addWeighted(frame0, 1.0 - t, frame1, t, 0.0)

