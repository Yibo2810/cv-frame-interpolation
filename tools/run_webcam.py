from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from vfi.backends import create_backend
from vfi.metrics import RuntimeStats, Stopwatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam interpolation demo.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", default="torch", choices=["torch", "rife", "flow", "blend", "compare"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--flow-preset", default="fast", choices=["ultrafast", "fast", "medium"])
    parser.add_argument("--rife-root", default="external/ECCV2022-RIFE")
    parser.add_argument("--rife-model-dir", default=None)
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--rife-tta", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--record-output", default=None)
    parser.add_argument("--record-fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited.")
    return parser.parse_args()


def make_backend(args: argparse.Namespace, name: str):
    return create_backend(
        name,
        device=args.device,
        weights=args.weights,
        preset=args.flow_preset,
        rife_root=args.rife_root,
        rife_model_dir=args.rife_model_dir,
        rife_scale=args.rife_scale,
        rife_tta=args.rife_tta,
    )


def make_record_path(path: str | None) -> Path:
    if path:
        record_path = Path(path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = Path("outputs") / f"webcam_recording_{timestamp}.mp4"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    return record_path


def draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out


def draw_status(frame: np.ndarray, recording: bool, record_path: Path | None) -> np.ndarray:
    if not recording:
        return frame
    out = frame.copy()
    cv2.circle(out, (24, 26), 8, (0, 0, 255), thickness=-1)
    text = "REC"
    if record_path is not None:
        text = f"REC {record_path.name}"
    cv2.putText(out, text, (42, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def is_window_visible(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def main() -> None:
    args = parse_args()
    if args.display_scale <= 0:
        raise ValueError("--display-scale must be greater than 0.")
    if args.record_fps <= 0:
        raise ValueError("--record-fps must be greater than 0.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.backend == "compare":
        backends = [("Flow", make_backend(args, "flow")), ("RIFE", make_backend(args, "rife"))]
    else:
        backends = [(args.backend.upper(), make_backend(args, args.backend))]

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first webcam frame.")
    prev = cv2.resize(prev, (args.width, args.height), interpolation=cv2.INTER_AREA)

    window_name = "Flow | RIFE" if args.backend == "compare" else "prev | interpolated | current"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    stats = RuntimeStats()
    backend_stats = {label: RuntimeStats() for label, _ in backends}
    frame_count = 1
    last_report = time.perf_counter()
    video_writer: cv2.VideoWriter | None = None
    record_path: Path | None = None

    while True:
        ok, curr = cap.read()
        if not ok:
            break
        curr = cv2.resize(curr, (args.width, args.height), interpolation=cv2.INTER_AREA)

        with Stopwatch() as total_timer:
            mids = []
            for label, backend in backends:
                with Stopwatch() as backend_timer:
                    mid = backend.interpolate(prev, curr, t=0.5)
                backend_stats[label].add_seconds(backend_timer.elapsed)
                mids.append(draw_label(mid, f"{label}: {backend_timer.elapsed * 1000.0:.1f} ms"))
        stats.add_seconds(total_timer.elapsed)

        if args.backend == "compare":
            display = np.hstack(mids)
        else:
            display = np.hstack([draw_label(prev, "Previous"), mids[0], draw_label(curr, "Current")])
        if abs(args.display_scale - 1.0) > 1e-6:
            display = cv2.resize(
                display,
                None,
                fx=args.display_scale,
                fy=args.display_scale,
                interpolation=cv2.INTER_AREA,
            )

        display = draw_status(display, video_writer is not None, record_path)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            if video_writer is None:
                record_path = make_record_path(args.record_output)
                height, width = display.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(record_path), fourcc, args.record_fps, (width, height))
                if not video_writer.isOpened():
                    video_writer.release()
                    video_writer = None
                    raise RuntimeError(f"Could not open video writer for {record_path}.")
                print(f"Recording started: {record_path}")
            else:
                video_writer.release()
                video_writer = None
                print(f"Recording stopped: {record_path}")
        if video_writer is not None:
            video_writer.write(display)

        if key in {ord("q"), 27}:
            break
        if not is_window_visible(window_name):
            break

        frame_count += 1
        prev = curr
        if args.max_frames and frame_count >= args.max_frames:
            break

        now = time.perf_counter()
        if now - last_report > 2.0:
            print(
                json.dumps(
                    {
                        "frames": frame_count,
                        "runtime": stats.summary(),
                        "backend_runtime": {label: stat.summary() for label, stat in backend_stats.items()},
                    }
                )
            )
            last_report = now

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print(
        json.dumps(
            {
                "backend": args.backend,
                "frames": frame_count,
                "runtime": stats.summary(),
                "backend_runtime": {label: stat.summary() for label, stat in backend_stats.items()},
                "recording": str(record_path) if record_path is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
