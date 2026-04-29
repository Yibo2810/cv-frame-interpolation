from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from vfi.backends import create_backend
from vfi.frame import write_image
from vfi.metrics import Stopwatch


def make_test_pair(width: int = 320, height: int = 180) -> tuple[np.ndarray, np.ndarray]:
    frame0 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1 = np.zeros_like(frame0)

    cv2.rectangle(frame0, (40, 55), (115, 125), (40, 180, 255), thickness=-1)
    cv2.rectangle(frame1, (82, 55), (157, 125), (40, 180, 255), thickness=-1)
    cv2.circle(frame0, (225, 90), 28, (255, 160, 30), thickness=-1)
    cv2.circle(frame1, (205, 90), 28, (255, 160, 30), thickness=-1)
    cv2.putText(frame0, "t", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    cv2.putText(frame1, "t+1", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    return frame0, frame1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic backend smoke test.")
    parser.add_argument("--backend", default="torch", choices=["torch", "rife", "flow", "blend"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flow-preset", default="fast", choices=["ultrafast", "fast", "medium"])
    parser.add_argument("--rife-root", default="external/ECCV2022-RIFE")
    parser.add_argument("--rife-model-dir", default=None)
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--rife-tta", action="store_true")
    parser.add_argument("--output-dir", default="outputs/smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame0, frame1 = make_test_pair()
    backend = create_backend(
        args.backend,
        device=args.device,
        preset=args.flow_preset,
        rife_root=args.rife_root,
        rife_model_dir=args.rife_model_dir,
        rife_scale=args.rife_scale,
        rife_tta=args.rife_tta,
    )

    with Stopwatch() as timer:
        mid = backend.interpolate(frame0, frame1, t=0.5)

    if mid.shape != frame0.shape:
        raise AssertionError(f"Expected output shape {frame0.shape}, got {mid.shape}.")
    if mid.dtype != np.uint8:
        raise AssertionError(f"Expected uint8 output, got {mid.dtype}.")

    write_image(str(output_dir / "frame0.png"), frame0)
    write_image(str(output_dir / "frame1.png"), frame1)
    write_image(str(output_dir / "middle.png"), mid)

    print(
        json.dumps(
            {
                "backend": args.backend,
                "output_dir": str(output_dir),
                "elapsed_ms": timer.elapsed * 1000.0,
                "shape": list(mid.shape),
                "dtype": str(mid.dtype),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
