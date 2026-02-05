"""NLP ports - Abstractions for entity extraction and intent classification.

These protocols define the contracts for NLP operations, allowing
multiple implementations (rule-based, SpaCy, HuggingFace) to be
swapped without changing the application logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from ..domain.models import Intent, StationExtractionResult


class StationExtractorPort(Protocol):
    """Port for station extraction from text.

    Replaces: pipeline.py:33 (StationExtractor Callable type alias)

    Implementations:
    - adapters/nlp/rule_based.py (from nlp/extract_stations.py:207-246)
    - adapters/nlp/hf_ner_adapter.py (from nlp/hf_ner/ner.py:153-177)
    - adapters/nlp/spacy_adapter.py (from nlp/legacy_spacy/extractor.py:37-40)
    """

    def extract(self, sentence: str) -> StationExtractionResult:
        """Extract departure and arrival stations from text.

        Args:
            sentence: The input sentence to analyze.

        Returns:
            StationExtractionResult with departure, arrival, and metadata.
        """
        ...


class LocationExtractorPort(Protocol):
    """Port for raw location extraction (NER).

    Replaces direct calls to:
    - extract_locations_hf (nlp/hf_ner/ner.py:117-132)
    - extract_locations_spacy (nlp/legacy_spacy/extractor.py:14-22)

    This extracts location entities without resolving them to station codes.
    """

    def extract_locations(self, text: str) -> Sequence[str]:
        """Extract location entity mentions from text.

        Args:
            text: The input text to analyze.

        Returns:
            Sequence of location strings found in the text.
        """
        ...


class DateExtractorPort(Protocol):
    """Port for date extraction.

    Replaces direct calls to:
    - extract_dates_hf (nlp/hf_ner/ner.py:135-150)
    - extract_dates_eds (nlp/legacy_spacy/extractor.py:25-34)
    """

    def extract_dates(self, text: str) -> Sequence[str]:
        """Extract date mentions from text.

        Args:
            text: The input text to analyze.

        Returns:
            Sequence of date strings found in the text.
        """
        ...


class IntentClassifierPort(Protocol):
    """Port for intent classification.

    Wraps: nlp/intent.py:555-600 (detect_intent)

    Intent classification determines whether the user's input
    is a valid French travel request.
    """

    def classify(self, sentence: str) -> Intent:
        """Classify the intent of a sentence.

        Args:
            sentence: The input sentence to classify.

        Returns:
            Intent enum value (TRIP, NOT_TRIP, NOT_FRENCH, UNKNOWN).
        """
        ...
