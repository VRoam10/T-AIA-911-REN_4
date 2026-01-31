from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("whisper_app")

torch: Optional[Any] = None

_torch: Optional[Any] = None
try:
    import torch as _torch_module
    _torch = _torch_module
except Exception:
    pass

torch = _torch

# NVML comes from "nvidia-ml-py" but import name is "pynvml"
try:
    import pynvml as _pynvml  # type: ignore
    _NVML_OK = True
except Exception:
    _pynvml = None
    _NVML_OK = False

pynvml: Optional[Any] = _pynvml

_NVML_INITIALIZED = False


def cuda_available(device: str = "cuda") -> bool:
    if device != "cuda":
        return False
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def log_gpu_memory(prefix: str = "", device: str = "cuda") -> None:
    """Logs allocated/reserved VRAM via torch (if available)."""
    if not cuda_available(device):
        return

    assert torch is not None

    try:
        dev = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev) / (1024**2)
        reserved = torch.cuda.memory_reserved(dev) / (1024**2)
        name = torch.cuda.get_device_name(dev)
        logger.info(
            f"{prefix}🧠 GPU: {name} | allocated={allocated:.0f}MB reserved={reserved:.0f}MB"
        )
    except Exception as e:
        logger.debug(f"GPU memory log failed: {e}")


def _nvml_init_if_needed() -> None:
    global _NVML_INITIALIZED
    if not _NVML_OK or _NVML_INITIALIZED:
        return
    if pynvml is None:
        return

    try:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
    except Exception as e:
        _NVML_INITIALIZED = False
        logger.debug(f"NVML init failed: {e}")


def _to_str(x: Any) -> str:
    """NVML may return bytes or str depending on version/platform."""
    if x is None:
        return "Unknown"
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def get_gpu_live_stats(device: str = "cuda") -> str:
    """
    Returns a markdown string with utilization and VRAM (NVML).
    Requires: pip install nvidia-ml-py
    """
    if device != "cuda":
        return "🧵 GPU Live: device is set to CPU."

    if not _NVML_OK:
        return "🧵 GPU Live: install `nvidia-ml-py` to see utilization (`pip install nvidia-ml-py`)."

    _nvml_init_if_needed()
    if not _NVML_INITIALIZED:
        return "🧵 GPU Live: NVML init failed (check NVIDIA driver / NVML availability)."

    if pynvml is None:
        return "🧵 GPU Live: NVML module is not available."

    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = _to_str(pynvml.nvmlDeviceGetName(h))

        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)

        mem_used = mem.used / (1024**2)
        mem_total = mem.total / (1024**2)

        return (
            f"🧵 **GPU Live**: {name}\n\n"
            f"- Utilization: **{util.gpu}%**\n"
            f"- VRAM: **{mem_used:.0f}MB / {mem_total:.0f}MB**\n"
        )
    except Exception as e:
        return f"🧵 GPU Live: error reading stats: {e}"


def clear_torch_cuda_cache(device: str = "cuda") -> Optional[str]:
    """Frees PyTorch CUDA cached allocator blocks (helps reduce VRAM after clearing models)."""
    if not cuda_available(device):
        return None

    assert torch is not None

    try:
        torch.cuda.empty_cache()
        return "✅ torch.cuda.empty_cache() done."
    except Exception as e:
        return f"⚠️ torch.cuda.empty_cache() failed: {e}"
