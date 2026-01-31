"""ASR adapters - Implementations of ASRModelPort.

Available implementations:
- WhisperASRAdapter: Faster-Whisper based ASR
"""

from .whisper_adapter import WhisperASRAdapter

__all__ = ["WhisperASRAdapter"]
