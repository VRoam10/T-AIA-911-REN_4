"""Fine-tuned CamemBERT intent classifier adapter.

Uses a locally fine-tuned CamemBERT model for intent classification
(TRIP / NOT_TRIP / NOT_FRENCH) with lazy loading and caching.
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
class FineTunedIntentClassifier:
    """Fine-tuned CamemBERT intent classifier.

    This adapter implements IntentClassifierPort using a locally
    fine-tuned CamemBERT model for sequence classification.

    Attributes:
        model_path: Path to the fine-tuned model directory.
        cache: Cache for pipeline instances.
        confidence_threshold: Minimum confidence for non-UNKNOWN classification.
    """

    model_path: str = "training/models/intent-camembert"
    cache: CachePort[Any] = field(
        default_factory=lambda: InMemoryCache(name="finetuned_intent")
    )
    confidence_threshold: float = 0.5

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _get_classifier(self) -> Any:
        """Get or lazily load the fine-tuned classification pipeline.

        Returns:
            The transformers text-classification pipeline.

        Raises:
            IntentClassificationError: If model loading fails.
        """
        cache_key = f"finetuned_intent:{self.model_path}"

        def load_pipeline() -> Any:
            self._logger.info(
                "Loading fine-tuned intent classifier (lazy)",
                extra={"model_path": self.model_path},
            )
            try:
                from transformers import pipeline

                return pipeline(
                    "text-classification",
                    model=self.model_path,
                    return_all_scores=True,
                )
            except Exception as e:
                raise IntentClassificationError(
                    f"Failed to load fine-tuned intent model: {e}",
                    cause=e,
                )

        return self.cache.get_or_compute(cache_key, load_pipeline)

    def classify(self, sentence: str) -> Intent:
        """Classify intent using fine-tuned CamemBERT model.

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
            results = classifier(sentence)

            # results is a list of list of dicts: [[{label, score}, ...]]
            scores = results[0]

            # Find the top prediction
            top = max(scores, key=lambda x: x["score"])
            top_label = top["label"]
            top_score = top["score"]

            self._logger.debug(
                "Fine-tuned intent result",
                extra={
                    "label": top_label,
                    "score": top_score,
                    "sentence_length": len(sentence),
                },
            )

            if top_score < self.confidence_threshold:
                return Intent.UNKNOWN

            label_mapping = {
                "TRIP": Intent.TRIP,
                "NOT_TRIP": Intent.NOT_TRIP,
                "NOT_FRENCH": Intent.NOT_FRENCH,
            }
            return label_mapping.get(top_label, Intent.UNKNOWN)

        except IntentClassificationError:
            raise
        except Exception as e:
            self._logger.error(
                "Fine-tuned intent classification failed",
                extra={"error": str(e)},
            )
            raise IntentClassificationError(
                f"Fine-tuned intent classification failed: {e}",
                cause=e,
            )

    def unload(self) -> None:
        """Clear cached pipelines."""
        cleared = self.cache.clear()
        self._logger.info(
            "Fine-tuned intent pipeline unloaded", extra={"cleared": cleared}
        )
