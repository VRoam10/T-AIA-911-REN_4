"""Whisper ASR adapter with proper caching.

This adapter wraps the Faster-Whisper ASR from asr.py with:
- Proper cache key handling (FIXES the device fallback bug!)
- Configuration injection
- Better error handling

CRITICAL FIX: Cache key bug in asr.py:63-65
BEFORE: Cache key uses requested device, but model may be on different device
AFTER: Cache key uses ACTUAL device the model was loaded on
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

from ...config import ASRConfig, get_config
from ...domain.errors import ASRError
from ...domain.models import TranscriptionResult, TranscriptionSegment
from ...ports.cache import CachePort
from ..cache.memory_cache import InMemoryCache


@dataclass
class WhisperASRAdapter:
    """Faster-Whisper ASR adapter with proper caching.

    This adapter implements ASRModelPort using Faster-Whisper.

    IMPORTANT: This fixes the cache key bug in asr.py:63-65 where the
    cache key used the requested device instead of the actual device
    after fallback.

    Attributes:
        config: ASR configuration
        cache: Cache for model instances
    """

    config: ASRConfig = field(default_factory=lambda: get_config().asr)
    cache: CachePort[Any] = field(default_factory=lambda: InMemoryCache(name="whisper"))

    _model: Optional[Any] = field(default=None, repr=False)
    _actual_device: str = field(default="", repr=False)
    _actual_compute: str = field(default="", repr=False)
    _model_id: str = field(default="", repr=False)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _load_model(self, model_id: Optional[str] = None) -> Any:
        """Load or get cached Whisper model.

        CRITICAL FIX: Cache key uses actual device, not requested device.
        """
        model_id = model_id or self.config.default_model
        requested_device: str = self.config.device
        requested_compute: str = self.config.compute_type

        # Determine actual device
        if requested_device == "auto":
            requested_device = self._detect_device()

        # Check cache with requested key first
        cache_key = f"{model_id}:{requested_device}:{requested_compute}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._logger.debug("Model cache hit", extra={"key": cache_key})
            self._model = cached
            self._model_id = model_id
            self._actual_device = requested_device
            self._actual_compute = requested_compute
            return cached

        # Load model
        self._logger.info(
            "Loading Whisper model",
            extra={
                "model": model_id,
                "device": requested_device,
                "compute_type": requested_compute,
            },
        )

        from faster_whisper import WhisperModel

        actual_device: str = requested_device
        actual_compute: str = requested_compute

        try:
            model = WhisperModel(
                model_id,
                device=requested_device,
                compute_type=requested_compute,
            )
            self._logger.info("Model loaded on requested device")
        except Exception as e:
            self._logger.warning(
                "Failed to load on requested device, falling back",
                extra={
                    "error": str(e),
                    "requested_device": requested_device,
                    "fallback_device": self.config.fallback_device,
                },
            )
            actual_device = self.config.fallback_device
            actual_compute = self.config.fallback_compute_type
            model = WhisperModel(
                model_id,
                device=actual_device,
                compute_type=actual_compute,
            )
            self._logger.info("Model loaded on fallback device")

        # CRITICAL FIX: Cache with ACTUAL device key, not requested
        actual_cache_key = f"{model_id}:{actual_device}:{actual_compute}"
        self.cache.set(actual_cache_key, model)

        self._model = model
        self._model_id = model_id
        self._actual_device = actual_device
        self._actual_compute = actual_compute

        return model

    def _detect_device(self) -> str:
        """Detect available device (cuda or cpu)."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.
            language: Language code hint (None for auto-detection).
            beam_size: Beam size for decoding.

        Returns:
            TranscriptionResult with full text and segments.

        Raises:
            ASRError: If transcription fails.
        """
        model = self._load_model()

        self._logger.info(
            "Starting transcription",
            extra={
                "audio_path": str(audio_path),
                "language": language,
                "beam_size": beam_size,
            },
        )

        start_time = time.time()

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
            )

            segments = []
            full_text_parts = []

            for segment in segments_iter:
                segments.append(
                    TranscriptionSegment(
                        start_seconds=segment.start,
                        end_seconds=segment.end,
                        text=segment.text.strip(),
                    )
                )
                full_text_parts.append(segment.text.strip())

            elapsed = time.time() - start_time

            result = TranscriptionResult(
                full_text=" ".join(full_text_parts),
                segments=tuple(segments),
                language=info.language or "unknown",
                language_probability=info.language_probability or 0.0,
                duration_seconds=info.duration,
            )

            self._logger.info(
                "Transcription complete",
                extra={
                    "elapsed_seconds": round(elapsed, 2),
                    "segments": len(segments),
                    "language": result.language,
                },
            )

            return result

        except Exception as e:
            self._logger.error(
                "Transcription failed",
                extra={"error": str(e), "audio_path": str(audio_path)},
            )
            raise ASRError(
                f"Transcription failed: {e}",
                model_id=self._model_id,
                device=self._actual_device,
                audio_path=str(audio_path),
                cause=e,
            )

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id or self.config.default_model

    @property
    def device(self) -> str:
        """Return the device the model is running on."""
        return self._actual_device or "unknown"

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            self._model = None
            self._logger.info("Whisper model unloaded")
