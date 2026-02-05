"""Intent classification adapter.

This adapter wraps the existing rule-based intent detection logic
from nlp/intent.py with the IntentClassifierPort interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ...domain.models import Intent


@dataclass
class RuleBasedIntentClassifier:
    """Rule-based intent classifier.

    This adapter implements IntentClassifierPort by wrapping the existing
    intent detection logic from nlp/intent.py.
    """

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def classify(self, sentence: str) -> Intent:
        """Classify the intent of a sentence.

        Args:
            sentence: The input sentence to classify.

        Returns:
            Intent enum value (TRIP, NOT_TRIP, NOT_FRENCH, UNKNOWN).
        """
        # Import here to avoid circular imports and leverage existing implementation
        from ...nlp.intent import Intent as LegacyIntent
        from ...nlp.intent import detect_intent as legacy_detect_intent

        legacy_intent = legacy_detect_intent(sentence)

        # Map legacy Intent to domain Intent
        intent_mapping = {
            LegacyIntent.TRIP: Intent.TRIP,
            LegacyIntent.NOT_TRIP: Intent.NOT_TRIP,
            LegacyIntent.NOT_FRENCH: Intent.NOT_FRENCH,
            LegacyIntent.UNKNOWN: Intent.UNKNOWN,
        }

        result = intent_mapping.get(legacy_intent, Intent.UNKNOWN)

        self._logger.debug(
            "Intent classified",
            extra={"sentence_length": len(sentence), "intent": result.name},
        )

        return result
