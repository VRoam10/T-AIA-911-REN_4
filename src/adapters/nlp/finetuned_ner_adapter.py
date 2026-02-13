"""Fine-tuned CamemBERT NER adapter for station extraction.

Uses a locally fine-tuned CamemBERT model with BIO tagging
(B-DEP, I-DEP, B-ARR, I-ARR, O) to extract departure and arrival
stations from text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ...config import NLPConfig, get_config
from ...domain.errors import ExtractionError
from ...domain.models import StationExtractionResult
from ...ports.cache import CachePort
from ..cache.memory_cache import InMemoryCache


@dataclass
class FineTunedNERAdapter:
    """Fine-tuned CamemBERT NER adapter for station extraction.

    This adapter implements StationExtractorPort using a locally
    fine-tuned CamemBERT token classification model with BIO tags.

    Attributes:
        model_path: Path to the fine-tuned NER model directory.
        config: NLP configuration.
        cache: Cache for pipeline instances.
    """

    model_path: str = "training/models/ner-camembert"
    config: NLPConfig = field(default_factory=lambda: get_config().nlp)
    cache: CachePort[Any] = field(
        default_factory=lambda: InMemoryCache(name="finetuned_ner")
    )

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _get_ner_pipeline(self) -> Any:
        """Get or lazily load the fine-tuned NER pipeline.

        Returns:
            The transformers token-classification pipeline.

        Raises:
            ExtractionError: If model loading fails.
        """
        cache_key = f"finetuned_ner:{self.model_path}"

        def load_pipeline() -> Any:
            self._logger.info(
                "Loading fine-tuned NER pipeline (lazy)",
                extra={"model_path": self.model_path},
            )
            try:
                from transformers import pipeline

                return pipeline(
                    "token-classification",
                    model=self.model_path,
                    aggregation_strategy="first",
                )
            except Exception as e:
                raise ExtractionError(
                    f"Failed to load fine-tuned NER model: {e}",
                    strategy_used="finetuned_ner",
                    cause=e,
                )

        return self.cache.get_or_compute(cache_key, load_pipeline)

    def _extract_entities_from_bio(self, sentence: str) -> tuple[list[str], list[str]]:
        """Run NER pipeline and extract DEP/ARR entity strings.

        Returns:
            Tuple of (departure_locations, arrival_locations).
        """
        pipe = self._get_ner_pipeline()
        entities = pipe(sentence)

        dep_parts: list[str] = []
        arr_parts: list[str] = []

        for entity in entities:
            label = entity.get("entity_group", "")
            word = entity.get("word", "").strip()
            word = re.sub(r"\s+", " ", word)

            if not word:
                continue

            if label == "DEP":
                dep_parts.append(word)
            elif label == "ARR":
                arr_parts.append(word)

        return dep_parts, arr_parts

    def extract(self, sentence: str) -> StationExtractionResult:
        """Extract departure and arrival stations using fine-tuned NER.

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
            dep_parts, arr_parts = self._extract_entities_from_bio(sentence)

            # Combine parts into full location names
            all_locations = dep_parts + arr_parts
            dep_name = " ".join(dep_parts) if dep_parts else None
            arr_name = " ".join(arr_parts) if arr_parts else None

            if not all_locations:
                return StationExtractionResult(
                    departure=None,
                    arrival=None,
                    raw_locations=(),
                    error="No locations detected.",
                )

            # Map locations to station codes
            from ...nlp.extract_stations import extract_stations_from_locations

            locations_for_mapping = []
            if dep_name:
                locations_for_mapping.append(dep_name)
            if arr_name:
                locations_for_mapping.append(arr_name)

            legacy_result = extract_stations_from_locations(
                sentence, locations_for_mapping
            )

            is_complete = (
                legacy_result.departure is not None
                and legacy_result.arrival is not None
            )

            result = StationExtractionResult(
                departure=legacy_result.departure,
                arrival=legacy_result.arrival,
                raw_locations=tuple(all_locations),
                confidence=1.0 if is_complete else 0.5,
                error=legacy_result.error,
            )

            self._logger.debug(
                "Station extraction (fine-tuned NER)",
                extra={
                    "departure": result.departure,
                    "arrival": result.arrival,
                    "dep_parts": dep_parts,
                    "arr_parts": arr_parts,
                    "success": result.is_success,
                },
            )

            return result

        except ExtractionError:
            raise
        except Exception as e:
            self._logger.error(
                "Fine-tuned NER extraction failed",
                extra={"error": str(e)},
            )
            raise ExtractionError(
                f"Fine-tuned NER extraction failed: {e}",
                strategy_used="finetuned_ner",
                cause=e,
            )

    def extract_locations(self, text: str) -> Sequence[str]:
        """Extract location entities from text.

        Args:
            text: The input text to analyze.

        Returns:
            Sequence of location strings found in the text.
        """
        dep_parts, arr_parts = self._extract_entities_from_bio(text)
        locations = []
        if dep_parts:
            locations.append(" ".join(dep_parts))
        if arr_parts:
            locations.append(" ".join(arr_parts))
        return locations

    def unload(self) -> None:
        """Clear cached pipelines."""
        cleared = self.cache.clear()
        self._logger.info(
            "Fine-tuned NER pipeline unloaded", extra={"cleared": cleared}
        )
