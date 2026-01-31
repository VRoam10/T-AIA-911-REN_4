"""ASR port - Abstraction for speech-to-text services.

This protocol defines the contract for ASR models, allowing
different implementations (Whisper, etc.) to be used.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from ..domain.models import TranscriptionResult


class ASRModelPort(Protocol):
    """Port for ASR models.

    Replaces: asr.py:49-66 (get_model) and asr.py:104-151 (transcribe_with_progress)
    Implementation: adapters/asr/whisper_adapter.py

    ASR models convert audio files to text transcriptions.
    """

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
            beam_size: Beam size for decoding (higher = more accurate, slower).

        Returns:
            TranscriptionResult with full text and segments.
        """
        ...

    @property
    def model_id(self) -> str:
        """Return the model identifier.

        Returns:
            The model ID string (e.g., 'large-v3').
        """
        ...

    @property
    def device(self) -> str:
        """Return the device the model is running on.

        Returns:
            Device string ('cuda' or 'cpu').
        """
        ...

    def unload(self) -> None:
        """Unload the model from memory.

        Call this to free GPU/CPU memory when the model is no longer needed.
        """
        ...
