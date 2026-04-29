# Implementation Plan

## Goal

Build a real-time webcam frame interpolation system that compares a pretrained
deep learning backend against a classical optical-flow backend under practical
latency constraints.

## Completed in v0.3.0

1. Backend interface and baseline structure.
   - Input: two consecutive `uint8` BGR frames.
   - Output: one synthesized intermediate `uint8` BGR frame.
   - Common parameter: interpolation time `t`, default `0.5`.

2. Classical computer vision baseline.
   - Uses OpenCV DIS dense optical flow.
   - Backward-warps both frames toward the midpoint.
   - Blends the warped frames.
   - Uses photometric disagreement as a simple fallback for occlusion and
     failure regions.

3. Pretrained RIFE backend.
   - Loads a local `external/ECCV2022-RIFE` checkout.
   - Loads pretrained v3.6 weights from `train_log/flownet.pkl`.
   - Runs inference through PyTorch.
   - Uses CUDA on the Windows RTX 4060 Laptop GPU when `--device cuda` or
     `--device auto` selects it.

4. Windows CUDA runtime setup.
   - Project-local `.venv` is the default runtime.
   - PyTorch CUDA, OpenCV, NumPy, and TorchVision are installed in the local
     environment.
   - Device check reports CUDA availability and selected device.
   - Flow, Torch, and RIFE smoke tests pass on Windows.

5. Realtime webcam demo.
   - Supports single-backend Flow, RIFE, Torch, and Blend modes.
   - Supports Flow-vs-RIFE side-by-side comparison through `--backend compare`.
   - Shows per-backend runtime overlays in the OpenCV preview.
   - Supports display scaling for wide preview layouts.
   - Exits cleanly through `q`, `Esc`, or the window close button.
   - Records the displayed preview to MP4 with the `r` key.

## Evaluation Plan

- Measure runtime per interpolated frame.
- Report effective FPS for Flow, RIFE, and Flow-vs-RIFE comparison mode.
- Test at 540p first, then 720p.
- Use Flow as the fast interpretable baseline.
- Use RIFE as the pretrained deep model quality comparison.
- Save MP4 recordings for presentation and course evidence.
- Discuss failure cases: occlusion, fast hand motion, blur, low light, and
  motion near object boundaries.

## Remaining Work

- Capture stable benchmark numbers for 540p and 720p.
- Choose a short recorded example for the final report.
- Add report figures comparing quality and runtime.
- Decide whether to keep the full RIFE preview in realtime mode or use a lower
  preview scale/resolution for smoother classroom demonstration.
