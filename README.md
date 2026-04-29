# Real-Time Webcam Frame Interpolation

Version: `v0.3.0-realtime-cuda-demo`

This project implements a real-time webcam frame interpolation pipeline for a
computer vision course project. It compares a classical OpenCV optical-flow
baseline against pretrained RIFE inference, with a Windows CUDA setup for live
webcam demos and video recording.

## Current Status

- Repository status: private course-project implementation.
- Package version: `0.3.0`.
- Implemented: Windows project-local `.venv` workflow with CUDA PyTorch.
- Implemented: RTX 4060 Laptop GPU detection through `torch.cuda`.
- Implemented: OpenCV DIS optical-flow backend for an interpretable CV baseline.
- Implemented: external RIFE adapter using `external/ECCV2022-RIFE` and v3.6
  pretrained weights.
- Implemented: pair-image, video-file, webcam, smoke-test, device-check, and
  RIFE setup-check scripts.
- Implemented: webcam window lifecycle fixes, display scaling, Flow-vs-RIFE
  comparison mode, runtime overlays, and keyboard-controlled MP4 recording.
- Not yet selected: public license. Keep this repository private until course
  and dependency/licensing requirements are clear.

## Project Layout

```text
CVproject/
|-- CHANGELOG.md
|-- VERSION
|-- docs/
|   |-- implementation_plan.md
|   |-- references.md
|   `-- rife_notes.md
|-- src/vfi/
|   |-- backends/
|   |   |-- base.py
|   |   |-- factory.py
|   |   |-- linear.py
|   |   |-- optical_flow.py
|   |   |-- rife_external.py
|   |   `-- torch_lite.py
|   |-- device.py
|   |-- frame.py
|   `-- metrics.py
`-- tools/
    |-- check_device.py
    |-- check_rife_setup.py
    |-- make_test_video.py
    |-- run_pair.py
    |-- run_video.py
    |-- run_webcam.py
    `-- smoke_test.py
```

The `external/`, `outputs/`, pretrained model weights, downloaded archives, and
generated media files are intentionally ignored by Git.

## Windows Quick Start

From PowerShell:

```powershell
cd E:\github_vault\cv-frame-interpolation
& .\.venv\Scripts\Activate.ps1
```

Check the selected PyTorch device:

```powershell
python tools\check_device.py
```

Expected Windows CUDA result:

```json
{
  "cuda_available": true,
  "selected_auto_device": "cuda"
}
```

Run smoke tests:

```powershell
python tools\smoke_test.py --backend flow --flow-preset ultrafast --output-dir outputs\smoke_flow
python tools\smoke_test.py --backend rife --device cuda --output-dir outputs\smoke_rife_cuda
python tools\smoke_test.py --backend torch --output-dir outputs\smoke_torch
```

## Webcam Demo

Flow 540p:

```powershell
python tools\run_webcam.py --backend flow --flow-preset ultrafast --width 960 --height 540
```

RIFE 540p on CUDA:

```powershell
python tools\run_webcam.py --backend rife --device cuda --width 960 --height 540 --display-scale 0.5
```

Flow vs RIFE side-by-side at 540p:

```powershell
python tools\run_webcam.py --backend compare --device cuda --flow-preset ultrafast --width 960 --height 540 --display-scale 0.5 --record-fps 15
```

720p comparison is heavier because each webcam frame runs both backends:

```powershell
python tools\run_webcam.py --backend compare --device cuda --flow-preset ultrafast --width 1280 --height 720 --display-scale 0.35 --record-fps 10 --record-output outputs\flow_vs_rife_720p.mp4
```

OpenCV window controls:

- `r`: start or stop MP4 recording.
- `q` or `Esc`: exit.
- Window close button: exit.

If no recording path is provided, recordings are saved under `outputs/` with a
timestamped filename such as `webcam_recording_YYYYMMDD_HHMMSS.mp4`.

## Backend Notes

- `flow`: OpenCV DIS optical flow baseline. It estimates dense motion, warps
  the two input frames toward the midpoint, blends them, and uses a simple
  photometric fallback for difficult regions.
- `rife`: External pretrained RIFE backend. It loads a local checkout from
  `external/ECCV2022-RIFE` and pretrained weights from `train_log/flownet.pkl`.
  On the Windows laptop it runs through CUDA PyTorch.
- `torch`: Lightweight PyTorch validation path using `TinyVFI`. It is useful
  for checking tensor/device plumbing but is not the report-grade RIFE model.
- `blend`: Linear blending baseline for sanity checks.
- `compare`: Webcam-only display mode that runs Flow and RIFE on the same frame
  pair and shows their interpolated outputs side by side with timing overlays.

## Version Notes

`v0.3.0-realtime-cuda-demo` moves the project from a Mac-oriented RIFE adapter
prototype to a Windows CUDA realtime demo. It adds a working CUDA PyTorch
environment, RIFE CUDA smoke testing, webcam display fixes, scaled live preview,
side-by-side Flow/RIFE comparison, and keyboard-controlled video recording for
course presentation evidence.
