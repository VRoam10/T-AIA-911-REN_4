"""SpaCy NER adapter with LAZY LOADING.

CRITICAL FIX: This adapter fixes the import-time model loading issue
in nlp/legacy_spacy/extractor.py:10-11.

BEFORE (problematic):
    nlp = spacy.load("fr_core_news_md")  # Loaded at import!

AFTER (this adapter):
    Model is loaded lazily on first use via _get_nlp()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from ...config import NLPConfig, get_config
from ...domain.models import StationExtractionResult
from ...ports.cache import CachePort
from ..cache.memory_cache import InMemoryCache

if TYPE_CHECKING:
    import spacy


@dataclass
class SpaCyNERAdapter:
    """SpaCy-based NER adapter with lazy model loading.

    This adapter implements StationExtractorPort and LocationExtractorPort
    with proper lazy loading of the SpaCy model.

    IMPORTANT: Unlike the legacy implementation, the SpaCy model is NOT
    loaded at import time. It is loaded on first use.

    Attributes:
        config: NLP configuration
        cache: Optional cache for model instances
    """

    config: NLPConfig = field(default_factory=lambda: get_config().nlp)
    cache: CachePort[Any] = field(default_factory=lambda: InMemoryCache(name="spacy"))

    _nlp: Optional[Any] = field(default=None, repr=False)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _get_nlp(self) -> Any:
        """Get or lazily load the SpaCy model.

        This method replaces the import-time loading in legacy_spacy/extractor.py:10-11.
        """
        if self._nlp is not None:
            return self._nlp

        self._logger.info(
            "Loading SpaCy model (lazy)",
            extra={"model": self.config.spacy_model},
        )

        import spacy

        self._nlp = spacy.load(self.config.spacy_model)

        # Add eds.dates pipe if not already present
        if "eds.dates" not in self._nlp.pipe_names:
            try:
                self._nlp.add_pipe("eds.dates")
                self._logger.debug("Added eds.dates pipe to SpaCy model")
            except Exception as e:
                self._logger.warning(
                    "Could not add eds.dates pipe",
                    extra={"error": str(e)},
                )

        return self._nlp

    def extract(self, sentence: str) -> StationExtractionResult:
        """Extract departure and arrival stations from text using SpaCy NER.

        Args:
            sentence: The input sentence to analyze.

        Returns:
            StationExtractionResult with departure, arrival, and metadata.
        """
        if not sentence or not sentence.strip():
            return StationExtractionResult(
                departure=None,
                arrival=None,
                raw_locations=(),
                error="Empty sentence.",
            )

        # Extract locations using SpaCy NER
        locations = self.extract_locations(sentence)

        if not locations:
            return StationExtractionResult(
                departure=None,
                arrival=None,
                raw_locations=(),
                error="No locations detected.",
            )

        # Map locations to station codes using existing logic
        from ...nlp.extract_stations import extract_stations_from_locations

        legacy_result = extract_stations_from_locations(sentence, list(locations))

        # Check if extraction was complete
        is_complete = (
            legacy_result.departure is not None and legacy_result.arrival is not None
        )

        result = StationExtractionResult(
            departure=legacy_result.departure,
            arrival=legacy_result.arrival,
            raw_locations=tuple(locations),
            confidence=1.0 if is_complete else 0.5,
            error=legacy_result.error,
        )

        self._logger.debug(
            "Station extraction (SpaCy)",
            extra={
                "departure": result.departure,
                "arrival": result.arrival,
                "locations": locations,
                "success": result.is_success,
            },
        )

        return result

    def extract_locations(self, text: str) -> Sequence[str]:
        """Extract location entities from text using SpaCy NER.

        Args:
            text: The input text to analyze.

        Returns:
            Sequence of location strings found in the text.
        """
        nlp = self._get_nlp()
        doc = nlp(text)

        # Extract LOC and GPE entities
        locations = sorted(
            {ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE")}
        )

        self._logger.debug(
            "Location extraction (SpaCy)",
            extra={"text_length": len(text), "locations": locations},
        )

        return locations

    def extract_dates(self, text: str) -> Sequence[str]:
        """Extract date expressions using eds.dates pipeline.

        Args:
            text: The input text to analyze.

        Returns:
            Sequence of date strings found in the text.
        """
        nlp = self._get_nlp()
        doc = nlp(text)

        # Get dates from eds.dates spans
        spans = doc.spans.get("dates", [])
        dates = [sp.text for sp in spans]

        self._logger.debug(
            "Date extraction (SpaCy/eds)",
            extra={"text_length": len(text), "dates": dates},
        )

        return dates

    def unload(self) -> None:
        """Unload the SpaCy model from memory."""
        if self._nlp is not None:
            self._nlp = None
            self._logger.info("SpaCy model unloaded")
