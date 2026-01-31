"""HuggingFace Intent Classifier adapter using zero-shot classification.

This adapter uses xlm-roberta-large-xnli for intent classification
with lazy loading and caching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ...domain.errors import IntentClassificationError
from ...domain.models import Intent
from ...ports.cache import CachePort
from ..cache.memory_cache import InMemoryCache


@dataclass
class HuggingFaceIntentClassifier:
    """Zero-shot intent classifier using xlm-roberta-large-xnli.

    This adapter implements IntentClassifierPort using HuggingFace
    zero-shot classification pipeline.

    Attributes:
        model_id: HuggingFace model identifier
        cache: Cache for pipeline instances
        confidence_threshold: Minimum confidence for non-UNKNOWN classification
    """

    model_id: str = "joeddav/xlm-roberta-large-xnli"
    cache: CachePort[Any] = field(
        default_factory=lambda: InMemoryCache(name="hf_intent")
    )
    confidence_threshold: float = 0.5

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _get_classifier(self) -> Any:
        """Get or lazily load the zero-shot classification pipeline.

        Returns:
            The transformers zero-shot classification pipeline.

        Raises:
            IntentClassificationError: If model loading fails.
        """
        cache_key = f"classifier:{self.model_id}"

        def load_pipeline() -> Any:
            self._logger.info(
                "Loading HuggingFace intent classifier (lazy)",
                extra={"model": self.model_id},
            )
            try:
                from transformers import pipeline

                return pipeline("zero-shot-classification", model=self.model_id)
            except Exception as e:
                raise IntentClassificationError(
                    f"Failed to load HF intent model: {e}",
                    cause=e,
                )

        return self.cache.get_or_compute(cache_key, load_pipeline)

    def classify(self, sentence: str) -> Intent:
        """Classify intent using zero-shot classification.

        Args:
            sentence: The input sentence to classify.

        Returns:
            Intent enum value (TRIP, NOT_TRIP, NOT_FRENCH, UNKNOWN).

        Raises:
            IntentClassificationError: If classification fails.
        """
        if not sentence or not sentence.strip():
            return Intent.UNKNOWN

        try:
            classifier = self._get_classifier()
            labels = ["demande de voyage", "autre demande", "texte non français"]

            result = classifier(
                sentence, labels, hypothesis_template="Ce texte est une {}."
            )

            top_label = result["labels"][0]
            top_score = result["scores"][0]

            self._logger.debug(
                "Classification result",
                extra={
                    "label": top_label,
                    "score": top_score,
                    "sentence_length": len(sentence),
                },
            )

            if top_score < self.confidence_threshold:
                return Intent.UNKNOWN

            mapping = {
                "demande de voyage": Intent.TRIP,
                "autre demande": Intent.NOT_TRIP,
                "texte non français": Intent.NOT_FRENCH,
            }
            return mapping.get(top_label, Intent.UNKNOWN)

        except IntentClassificationError:
            raise
        except Exception as e:
            self._logger.error(
                "HF intent classification failed",
                extra={"error": str(e)},
            )
            raise IntentClassificationError(
                f"HF intent classification failed: {e}",
                cause=e,
            )

    def unload(self) -> None:
        """Clear cached pipelines."""
        cleared = self.cache.clear()
        self._logger.info("HF intent pipeline unloaded", extra={"cleared": cleared})
