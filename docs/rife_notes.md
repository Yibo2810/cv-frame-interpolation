# RIFE Notes

## Is RIFE AI?

Yes. RIFE is a deep learning model for video frame interpolation. In this
project, "using RIFE" means running inference with a pretrained neural network,
not training a new model from scratch.

The normal workflow is:

1. Download or clone the RIFE implementation.
2. Download pretrained model weights.
3. Feed two frames into the network.
4. Run a forward pass on CUDA when available.
5. Receive the synthesized middle frame.

For this course project, the contribution is the real-time system integration,
backend comparison, latency measurement, recording workflow, and failure
analysis. The RIFE backend should be described honestly as pretrained deep model
inference.

## What RIFE Predicts

Given two frames:

```text
I_t, I_{t+1}
```

RIFE estimates intermediate motion and blending information internally, then
generates:

```text
I_{t+0.5}
```

Unlike the classical baseline, we do not manually compute optical flow and masks
outside the model. The network learns how to estimate and combine motion cues
from training data.

## Difference From the Optical-Flow Baseline

Classical backend:

```text
OpenCV DIS flow -> warp both frames -> blend -> simple confidence fallback
```

RIFE backend:

```text
pretrained neural network -> learned intermediate flow/mask/features -> output
```

The classical baseline is easier to explain mathematically and is fast enough
for realtime preview. RIFE is expected to produce better visual quality around
complex motion, but it depends more heavily on GPU support and model
compatibility.

## Windows CUDA Status

The current v0.3.0 setup has moved beyond the earlier Mac-only development
assumption:

- The project-local Windows `.venv` contains CUDA PyTorch.
- `tools/check_device.py` reports `cuda_available: true`.
- The selected automatic device is `cuda`.
- The tested GPU is an NVIDIA GeForce RTX 4060 Laptop GPU.
- `tools/smoke_test.py --backend rife --device cuda` loads `RIFE_HDv3` and
  produces a valid interpolated frame.

Mac can still be useful for editing, project structure, and low-resolution
experiments. The Windows CUDA laptop is now the main target for the realtime
RIFE demo and course recording.

## Setup Expectation

The `rife` backend expects:

```text
CVproject/external/ECCV2022-RIFE/
CVproject/external/ECCV2022-RIFE/train_log/flownet.pkl
```

or another model directory passed with `--rife-model-dir`.

The `external/` directory, downloaded archives, and pretrained weight files are
intentionally ignored by Git. This repository tracks the adapter and project
code, not a vendored copy of the third-party RIFE implementation.

Current local status:

- Official RIFE code is cloned under `external/ECCV2022-RIFE`.
- The v3.6 pretrained package was downloaded from Hugging Face.
- The required `flownet.pkl`, `RIFE_HDv3.py`, and `IFNet_HDv3.py` files are in
  `external/ECCV2022-RIFE/train_log/`.
- CUDA smoke test passed on Windows.

## Realtime Demo Notes

Single-backend RIFE mode:

```powershell
python tools\run_webcam.py --backend rife --device cuda --width 960 --height 540 --display-scale 0.5
```

Flow-vs-RIFE comparison mode:

```powershell
python tools\run_webcam.py --backend compare --device cuda --flow-preset ultrafast --width 960 --height 540 --display-scale 0.5 --record-fps 15
```

Comparison mode is useful for presentation, but it is slower because each frame
pair runs both Flow and RIFE. Start with 540p; use 720p only if the live demo
remains responsive enough.
