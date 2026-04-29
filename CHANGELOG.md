# Changelog

## v0.3.0-realtime-cuda-demo - 2026-04-29

- Migrated the runnable project setup from a Mac-oriented workflow to a
  project-local Windows `.venv` workflow.
- Installed and validated CUDA PyTorch on the RTX 4060 Laptop GPU
  (`torch 2.11.0+cu128`, CUDA available through `torch.cuda`).
- Downloaded and validated the external ECCV2022-RIFE checkout plus v3.6
  pretrained `train_log/flownet.pkl` weights.
- Verified Flow, Torch, and RIFE smoke tests from the Windows environment.
- Updated the Makefile defaults so commands point at the project-local virtual
  environment instead of a hard-coded Mac Python path.
- Added `torchvision` to project requirements because the RIFE external code
  imports it during model setup.
- Fixed the OpenCV webcam window lifecycle so closing the window, pressing `q`,
  or pressing `Esc` stops the loop instead of recreating the window.
- Added display scaling for webcam output so 540p and 720p previews do not
  exceed the screen width.
- Added `compare` webcam mode to show Flow and RIFE interpolated outputs side
  by side with per-backend timing overlays.
- Added keyboard-controlled MP4 recording from the webcam preview with `r` to
  start or stop recording and optional output path/FPS settings.
- Rewrote README usage notes for Windows CUDA testing, realtime webcam demos,
  Flow-vs-RIFE comparison, and video recording.

Status: Windows CUDA realtime demo is runnable. Flow is suitable for fast live
baseline testing; RIFE provides the pretrained deep model comparison and is the
main performance bottleneck, especially in 720p side-by-side mode.

## v0.2.0-rife-adapter - 2026-04-28

- Added an external RIFE backend adapter for pretrained frame interpolation inference.
- Added CLI options for RIFE root, model directory, scale, and TTA settings.
- Added RIFE setup checking, synthetic test-video generation, and Makefile targets.
- Added RIFE notes explaining the pretrained model workflow and current Mac/CUDA expectations.

Status: adapter implemented and locally smoke-tested. The external RIFE repository,
downloaded archives, and pretrained weights are local dependencies and are not
vendored into this repository.

## v0.1.0-skeleton - 2026-04-28

- Added the initial project skeleton for real-time video frame interpolation.
- Added backend interfaces for torch-lite, OpenCV DIS optical flow, and linear blending.
- Added pair-image, video-file, webcam, device-check, and smoke-test scripts.
- Added project documentation and a near-term implementation roadmap.

Status: skeleton implementation. The pretrained RIFE backend and RTX4060 CUDA
benchmark are not implemented yet.
