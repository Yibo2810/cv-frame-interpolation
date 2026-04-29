from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small synthetic motion video for pipeline tests.")
    parser.add_argument("--output", default="outputs/test_motion.mp4")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output}")

    for i in range(args.frames):
        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        frame[:] = (12, 12, 18)

        phase = i / max(1, args.frames - 1)
        x0 = int(20 + phase * (args.width - 120))
        y0 = int(args.height * 0.35)
        cv2.rectangle(frame, (x0, y0), (x0 + 70, y0 + 50), (40, 180, 255), thickness=-1)

        cx = int(args.width * 0.75 - phase * args.width * 0.35)
        cy = int(args.height * 0.62)
        cv2.circle(frame, (cx, cy), 24, (255, 160, 30), thickness=-1)

        line_x = int(args.width * phase)
        cv2.line(frame, (line_x, 0), (args.width - line_x, args.height), (120, 220, 120), 3)
        cv2.putText(frame, f"{i:02d}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        writer.write(frame)

    writer.release()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

