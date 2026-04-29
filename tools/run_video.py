from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from _bootstrap import add_src_to_path

add_src_to_path()

from vfi.backends import create_backend
from vfi.frame import resize_if_needed
from vfi.metrics import RuntimeStats, Stopwatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2x interpolate a video file.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument("--backend", default="torch", choices=["torch", "rife", "flow", "blend"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--flow-preset", default="fast", choices=["ultrafast", "fast", "medium"])
    parser.add_argument("--rife-root", default="external/ECCV2022-RIFE")
    parser.add_argument("--rife-model-dir", default=None)
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--rife-tta", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 1e-3:
        input_fps = 30.0

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Input video has no frames.")
    prev = resize_if_needed(prev, args.width, args.height)
    height, width = prev.shape[:2]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, input_fps * 2.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {args.output}")

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
    stats = RuntimeStats()
    frame_count = 1
    writer.write(prev)

    while True:
        if args.max_frames is not None and frame_count >= args.max_frames:
            break
        ok, curr = cap.read()
        if not ok:
            break
        curr = resize_if_needed(curr, width, height)

        with Stopwatch() as timer:
            mid = backend.interpolate(prev, curr, t=0.5)
        stats.add_seconds(timer.elapsed)

        writer.write(mid)
        writer.write(curr)

        if args.display:
            cv2.imshow("interpolated mid", mid)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev = curr
        frame_count += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(
        json.dumps(
            {
                "backend": args.backend,
                "input": args.input,
                "output": args.output,
                "input_fps": input_fps,
                "output_fps": input_fps * 2.0,
                "input_frames_processed": frame_count,
                "interpolated_frames": max(0, frame_count - 1),
                "runtime": stats.summary(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
