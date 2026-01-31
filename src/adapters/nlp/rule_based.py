"""Rule-based station extractor adapter.

This adapter wraps the existing rule-based extraction logic
from nlp/extract_stations.py with the StationExtractorPort interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ...config import GraphConfig, get_config
from ...domain.models import StationExtractionResult


@dataclass
class RuleBasedStationExtractor:
    """Rule-based station extractor using CSV lookup.

    This adapter implements StationExtractorPort by wrapping the existing
    rule-based extraction logic from nlp/extract_stations.py.

    Attributes:
        config: Graph configuration for station CSV path
    """

    config: GraphConfig = field(default_factory=lambda: get_config().graph)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def extract(self, sentence: str) -> StationExtractionResult:
        """Extract departure and arrival stations from text.

        Args:
            sentence: The input sentence to analyze.

        Returns:
            StationExtractionResult with departure, arrival, and metadata.
        """
        # Import here to leverage existing implementation
        from ...nlp.extract_stations import (
            extract_stations as legacy_extract,
        )
        from ...nlp.extract_stations import (
            StationExtractionResult as LegacyResult,
        )

        legacy_result: LegacyResult = legacy_extract(sentence)

        # Check if extraction was complete
        is_complete = (
            legacy_result.departure is not None and legacy_result.arrival is not None
        )

        # Convert to domain model
        result = StationExtractionResult(
            departure=legacy_result.departure,
            arrival=legacy_result.arrival,
            raw_locations=(),  # Rule-based doesn't track raw locations
            confidence=1.0 if is_complete else 0.5,
            error=legacy_result.error,
        )

        self._logger.debug(
            "Station extraction (rule-based)",
            extra={
                "departure": result.departure,
                "arrival": result.arrival,
                "success": result.is_success,
            },
        )

        return result
