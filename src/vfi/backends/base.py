from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InterpolationBackend(ABC):
    name = "base"

    @abstractmethod
    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        """Return an intermediate frame between frame0 and frame1."""

