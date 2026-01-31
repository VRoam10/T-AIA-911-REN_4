"""HuggingFace NER adapter with lazy loading and caching.

This adapter wraps the HuggingFace NER functionality from nlp/hf_ner/ner.py
with proper dependency injection and lazy loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from ...config import NLPConfig, get_config
from ...domain.errors import ExtractionError
from ...domain.models import StationExtractionResult
from ...ports.cache import CachePort
from ..cache.memory_cache import InMemoryCache


@dataclass
class HuggingFaceNERAdapter:
    """HuggingFace-based NER adapter with lazy model loading.

    This adapter implements StationExtractorPort and LocationExtractorPort
    using HuggingFace transformers models.

    Attributes:
        config: NLP configuration
        cache: Cache for model instances
    """

    config: NLPConfig = field(default_factory=lambda: get_config().nlp)
    cache: CachePort[Any] = field(default_factory=lambda: InMemoryCache(name="hf_ner"))

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _get_ner_pipeline(self, model_id: Optional[str] = None) -> Any:
        """Get or lazily load the HuggingFace NER pipeline.

        Args:
            model_id: Optional model ID override.

        Returns:
            The transformers NER pipeline.
        """
        model_id = model_id or self.config.hf_ner_model
        cache_key = f"ner_pipeline:{model_id}"

        def load_pipeline() -> Any:
            self._logger.info(
                "Loading HuggingFace NER pipeline (lazy)",
                extra={"model": model_id},
            )
            try:
                from transformers import pipeline

                return pipeline(
                    "ner",
                    model=model_id,
                    aggregation_strategy="simple",
                )
            except Exception as e:
                raise ExtractionError(
                    f"Failed to load HF NER model: {e}",
                    strategy_used="hf_ner",
                    cause=e,
                )

        return self.cache.get_or_compute(cache_key, load_pipeline)

    def extract(self, sentence: str) -> StationExtractionResult:
        """Extract departure and arrival stations from text using HF NER.

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

        try:
            # Extract locations using HF NER
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
                legacy_result.departure is not None
                and legacy_result.arrival is not None
            )

            result = StationExtractionResult(
                departure=legacy_result.departure,
                arrival=legacy_result.arrival,
                raw_locations=tuple(locations),
                confidence=1.0 if is_complete else 0.5,
                error=legacy_result.error,
            )

            self._logger.debug(
                "Station extraction (HF NER)",
                extra={
                    "departure": result.departure,
                    "arrival": result.arrival,
                    "locations": locations,
                    "success": result.is_success,
                },
            )

            return result

        except ExtractionError:
            raise
        except Exception as e:
            self._logger.error(
                "HF NER extraction failed",
                extra={"error": str(e)},
            )
            raise ExtractionError(
                f"HF NER extraction failed: {e}",
                strategy_used="hf_ner",
                cause=e,
            )

    def extract_locations(self, text: str) -> Sequence[str]:
        """Extract location entities from text using HF NER.

        Args:
            text: The input text to analyze.

        Returns:
            Sequence of location strings found in the text.
        """
        pipe = self._get_ner_pipeline()
        entities = pipe(text)

        # Filter for location entities (LOC, GPE, etc.)
        location_labels = {"LOC", "GPE", "LOCATION", "I-LOC", "B-LOC"}
        locations = []

        for entity in entities:
            label = entity.get("entity_group") or entity.get("entity", "")
            if label in location_labels:
                word = entity.get("word", "").strip()
                if word and word not in locations:
                    locations.append(word)

        self._logger.debug(
            "Location extraction (HF NER)",
            extra={"text_length": len(text), "locations": locations},
        )

        return locations

    def extract_dates(self, text: str) -> Sequence[str]:
        """Extract date expressions using HF NER.

        Args:
            text: The input text to analyze.

        Returns:
            Sequence of date strings found in the text.
        """
        # Use dates-specific model if configured
        model_id = self.config.hf_ner_dates_model
        pipe = self._get_ner_pipeline(model_id)
        entities = pipe(text)

        # Filter for date entities
        date_labels = {"DATE", "TIME", "I-DATE", "B-DATE"}
        dates = []

        for entity in entities:
            label = entity.get("entity_group") or entity.get("entity", "")
            if label in date_labels:
                word = entity.get("word", "").strip()
                if word and word not in dates:
                    dates.append(word)

        self._logger.debug(
            "Date extraction (HF NER)",
            extra={"text_length": len(text), "dates": dates},
        )

        return dates

    def unload(self) -> None:
        """Clear cached pipelines."""
        cleared = self.cache.clear()
        self._logger.info("HF NER pipelines unloaded", extra={"cleared": cleared})
