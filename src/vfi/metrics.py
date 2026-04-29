from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class RuntimeStats:
    samples_ms: list[float] = field(default_factory=list)

    def add_seconds(self, seconds: float) -> None:
        self.samples_ms.append(seconds * 1000.0)

    def summary(self) -> dict[str, float | int]:
        if not self.samples_ms:
            return {"count": 0}
        mean_ms = statistics.fmean(self.samples_ms)
        return {
            "count": len(self.samples_ms),
            "mean_ms": mean_ms,
            "median_ms": statistics.median(self.samples_ms),
            "min_ms": min(self.samples_ms),
            "max_ms": max(self.samples_ms),
            "effective_fps": 1000.0 / mean_ms if mean_ms > 0.0 else 0.0,
        }


class Stopwatch:
    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start

