from __future__ import annotations

from vfi.backends.base import InterpolationBackend
from vfi.backends.linear import LinearBlendBackend
from vfi.backends.optical_flow import DISFlowBackend


def create_backend(name: str, **kwargs) -> InterpolationBackend:
    name = name.lower()
    if name in {"torch", "torch-lite", "tiny"}:
        from vfi.backends.torch_lite import TorchLiteBackend

        return TorchLiteBackend(
            device=kwargs.get("device", "auto"),
            weights=kwargs.get("weights"),
            compile_model=kwargs.get("compile_model", False),
        )
    if name in {"flow", "dis", "opencv"}:
        return DISFlowBackend(
            preset=kwargs.get("preset", "fast"),
            photometric_threshold=kwargs.get("photometric_threshold", 35.0),
        )
    if name in {"rife", "rife-external"}:
        from vfi.backends.rife_external import RIFEExternalBackend

        return RIFEExternalBackend(
            repo_path=kwargs.get("rife_root", "external/ECCV2022-RIFE"),
            model_dir=kwargs.get("rife_model_dir"),
            device=kwargs.get("device", "auto"),
            scale=kwargs.get("rife_scale", 1.0),
            tta=kwargs.get("rife_tta", False),
        )
    if name in {"blend", "linear"}:
        return LinearBlendBackend()
    raise ValueError("Unknown backend. Choose from: torch, rife, flow, blend.")
