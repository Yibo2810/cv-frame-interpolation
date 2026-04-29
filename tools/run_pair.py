from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from vfi.backends import create_backend
from vfi.frame import read_image, write_image
from vfi.metrics import Stopwatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate one middle frame from two images.")
    parser.add_argument("--img0", required=True, help="First input image.")
    parser.add_argument("--img1", required=True, help="Second input image.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--backend", default="torch", choices=["torch", "rife", "flow", "blend"])
    parser.add_argument("--device", default="auto", help="PyTorch device for torch backend.")
    parser.add_argument("--weights", default=None, help="Optional TinyVFI weights path.")
    parser.add_argument("--flow-preset", default="fast", choices=["ultrafast", "fast", "medium"])
    parser.add_argument("--rife-root", default="external/ECCV2022-RIFE")
    parser.add_argument("--rife-model-dir", default=None)
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--rife-tta", action="store_true")
    parser.add_argument("--t", type=float, default=0.5, help="Interpolation time in [0, 1].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = create_backend(
        args.backend,
        device=args.device,
        weights=args.weights,
        preset=args.flow_preset,
        rife_root=args.rife_root,
        rife_model_dir=args.rife_model_dir,
        rife_scale=args.rife_scale,
        rife_tta=args.rife_tta,
    )
    frame0 = read_image(args.img0)
    frame1 = read_image(args.img1)

    with Stopwatch() as timer:
        mid = backend.interpolate(frame0, frame1, t=args.t)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_image(args.output, mid)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "output": args.output,
                "elapsed_ms": timer.elapsed * 1000.0,
                "shape": list(mid.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
