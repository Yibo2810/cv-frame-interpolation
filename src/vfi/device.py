from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a PyTorch device for Mac-first development."""
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is False.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def describe_torch_devices() -> dict[str, object]:
    """Return a compact device report for scripts and logs."""
    return {
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_auto_device": str(resolve_device("auto")),
    }

