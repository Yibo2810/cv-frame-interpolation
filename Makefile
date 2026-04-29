ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
endif
SYSTEM_PYTHON ?= $(PYTHON)
IMG0 ?= outputs/smoke_torch/frame0.png
IMG1 ?= outputs/smoke_torch/frame1.png
OUT ?= outputs/mid.png

.PHONY: check-device check-rife smoke-torch smoke-rife smoke-rife-cuda smoke-flow smoke-blend test-video video-rife pair-torch pair-rife pair-flow webcam-torch webcam-rife webcam-rife-cuda webcam-flow webcam-compare

check-device:
	$(PYTHON) tools/check_device.py

check-rife:
	$(PYTHON) tools/check_rife_setup.py

smoke-torch:
	$(PYTHON) tools/smoke_test.py --backend torch --output-dir outputs/smoke_torch

smoke-rife:
	$(PYTHON) tools/smoke_test.py --backend rife --device cpu --output-dir outputs/smoke_rife_cpu

smoke-rife-cuda:
	$(PYTHON) tools/smoke_test.py --backend rife --device cuda --output-dir outputs/smoke_rife_cuda

smoke-flow:
	$(SYSTEM_PYTHON) tools/smoke_test.py --backend flow --flow-preset ultrafast --output-dir outputs/smoke_flow

smoke-blend:
	$(SYSTEM_PYTHON) tools/smoke_test.py --backend blend --output-dir outputs/smoke_blend

test-video:
	$(SYSTEM_PYTHON) tools/make_test_video.py --output outputs/test_motion.mp4 --width 320 --height 180 --frames 30 --fps 15

video-rife: test-video
	$(PYTHON) tools/run_video.py --input outputs/test_motion.mp4 --output outputs/test_motion_rife_2x.mp4 --backend rife --device cpu --max-frames 20

pair-torch:
	$(PYTHON) tools/run_pair.py --img0 $(IMG0) --img1 $(IMG1) --backend torch --output $(OUT)

pair-rife:
	$(PYTHON) tools/run_pair.py --img0 $(IMG0) --img1 $(IMG1) --backend rife --output $(OUT)

pair-flow:
	$(SYSTEM_PYTHON) tools/run_pair.py --img0 $(IMG0) --img1 $(IMG1) --backend flow --flow-preset ultrafast --output $(OUT)

webcam-torch:
	$(PYTHON) tools/run_webcam.py --backend torch --width 640 --height 360

webcam-rife:
	$(PYTHON) tools/run_webcam.py --backend rife --width 640 --height 360

webcam-rife-cuda:
	$(PYTHON) tools/run_webcam.py --backend rife --device cuda --width 960 --height 540 --display-scale 0.5

webcam-flow:
	$(SYSTEM_PYTHON) tools/run_webcam.py --backend flow --flow-preset ultrafast --width 640 --height 360

webcam-compare:
	$(PYTHON) tools/run_webcam.py --backend compare --device cuda --flow-preset ultrafast --width 960 --height 540 --display-scale 0.5 --record-fps 15
