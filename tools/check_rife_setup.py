from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local RIFE repo and weight layout.")
    parser.add_argument("--rife-root", default="external/ECCV2022-RIFE")
    parser.add_argument("--rife-model-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.rife_root).expanduser().resolve()
    model_dir = Path(args.rife_model_dir).expanduser().resolve() if args.rife_model_dir else root / "train_log"
    report = {
        "rife_root": str(root),
        "rife_root_exists": root.exists(),
        "inference_img_exists": (root / "inference_img.py").exists(),
        "model_dir": str(model_dir),
        "model_dir_exists": model_dir.exists(),
        "flownet_pkl_exists": (model_dir / "flownet.pkl").exists(),
        "rife_hdv3_py_exists": (model_dir / "RIFE_HDv3.py").exists(),
    }
    report["ready_for_backend"] = bool(
        report["rife_root_exists"]
        and report["inference_img_exists"]
        and report["model_dir_exists"]
        and (report["flownet_pkl_exists"] or report["rife_hdv3_py_exists"])
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

