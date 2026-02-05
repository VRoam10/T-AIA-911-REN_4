"""Extraction service with explicit fallback handling.

This service orchestrates NLP extraction with proper logging
when fallbacks occur, replacing the silent exception swallowing
in strategies.py:189-193 and 207-211.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..domain.errors import ExtractionError
from ..domain.models import City, ExtractionResult
from ..ports.geocoding import GeocoderPort
from ..ports.nlp import DateExtractorPort, LocationExtractorPort


@dataclass
class ExtractionService:
    """NLP extraction orchestration with explicit fallback handling.

    This service replaces the silent fallback logic in strategies.py
    with proper logging and error handling.

    Attributes:
        primary_location_extractor: Primary location extractor
        primary_date_extractor: Primary date extractor
        fallback_location_extractor: Optional fallback extractor
        fallback_date_extractor: Optional fallback extractor
        geocoder: Geocoder for validating locations
    """

    primary_location_extractor: LocationExtractorPort
    primary_date_extractor: DateExtractorPort
    fallback_location_extractor: Optional[LocationExtractorPort] = None
    fallback_date_extractor: Optional[DateExtractorPort] = None
    geocoder: Optional[GeocoderPort] = None

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def extract_locations(self, text: str) -> Sequence[str]:
        """Extract locations with logged fallback.

        Replaces: strategies.py:178-194 (extract_locations_by_strategy)

        Args:
            text: Input text to analyze.

        Returns:
            Sequence of location strings.

        Raises:
            ExtractionError: If all extractors fail.
        """
        primary_name = type(self.primary_location_extractor).__name__

        try:
            locations = self.primary_location_extractor.extract_locations(text)
            self._logger.debug(
                "Location extraction succeeded",
                extra={"extractor": primary_name, "locations": list(locations)},
            )
            return locations

        except Exception as e:
            if self.fallback_location_extractor is None:
                self._logger.error(
                    "Location extraction failed, no fallback available",
                    extra={"extractor": primary_name, "error": str(e)},
                )
                raise ExtractionError(
                    f"Location extraction failed: {e}",
                    strategy_used=primary_name,
                    fallback_attempted=False,
                    cause=e,
                )

            fallback_name = type(self.fallback_location_extractor).__name__

            # IMPORTANT: Log the fallback (fixes silent swallowing)
            self._logger.warning(
                "Primary location extractor failed, using fallback",
                extra={
                    "primary": primary_name,
                    "fallback": fallback_name,
                    "error": str(e),
                },
            )

            try:
                locations = self.fallback_location_extractor.extract_locations(text)
                self._logger.debug(
                    "Fallback location extraction succeeded",
                    extra={"extractor": fallback_name, "locations": list(locations)},
                )
                return locations

            except Exception as fallback_error:
                self._logger.error(
                    "Fallback location extraction also failed",
                    extra={
                        "primary": primary_name,
                        "fallback": fallback_name,
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error),
                    },
                )
                raise ExtractionError(
                    f"All location extractors failed",
                    strategy_used=fallback_name,
                    fallback_attempted=True,
                    cause=fallback_error,
                )

    def extract_dates(self, text: str) -> Sequence[str]:
        """Extract dates with logged fallback.

        Replaces: strategies.py:197-212 (extract_dates_by_strategy)

        Args:
            text: Input text to analyze.

        Returns:
            Sequence of date strings.

        Raises:
            ExtractionError: If all extractors fail.
        """
        primary_name = type(self.primary_date_extractor).__name__

        try:
            dates = self.primary_date_extractor.extract_dates(text)
            self._logger.debug(
                "Date extraction succeeded",
                extra={"extractor": primary_name, "dates": list(dates)},
            )
            return dates

        except Exception as e:
            if self.fallback_date_extractor is None:
                self._logger.error(
                    "Date extraction failed, no fallback available",
                    extra={"extractor": primary_name, "error": str(e)},
                )
                raise ExtractionError(
                    f"Date extraction failed: {e}",
                    strategy_used=primary_name,
                    fallback_attempted=False,
                    cause=e,
                )

            fallback_name = type(self.fallback_date_extractor).__name__

            # IMPORTANT: Log the fallback (fixes silent swallowing)
            self._logger.warning(
                "Primary date extractor failed, using fallback",
                extra={
                    "primary": primary_name,
                    "fallback": fallback_name,
                    "error": str(e),
                },
            )

            try:
                dates = self.fallback_date_extractor.extract_dates(text)
                self._logger.debug(
                    "Fallback date extraction succeeded",
                    extra={"extractor": fallback_name, "dates": list(dates)},
                )
                return dates

            except Exception as fallback_error:
                self._logger.error(
                    "Fallback date extraction also failed",
                    extra={
                        "primary": primary_name,
                        "fallback": fallback_name,
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error),
                    },
                )
                raise ExtractionError(
                    f"All date extractors failed",
                    strategy_used=fallback_name,
                    fallback_attempted=True,
                    cause=fallback_error,
                )

    def extract_and_validate_cities(self, text: str) -> Sequence[City]:
        """Extract locations and validate via geocoding.

        Args:
            text: Input text to analyze.

        Returns:
            Sequence of validated cities.
        """
        locations = self.extract_locations(text)

        if not locations:
            return []

        if self.geocoder is None:
            self._logger.warning("No geocoder configured, skipping validation")
            return []

        cities = []
        for loc in locations:
            city = self.geocoder.geocode(loc)
            if city is not None:
                cities.append(city)
            else:
                self._logger.debug(
                    "Location not geocodable",
                    extra={"location": loc},
                )

        return cities

    def run_full_extraction(
        self,
        text: str,
        normalize_dates: bool = True,
    ) -> ExtractionResult:
        """Run full extraction pipeline.

        Replaces: strategies.py:215-249 (run_extraction)

        Args:
            text: Input text to analyze.
            normalize_dates: Whether to normalize dates to ISO format.

        Returns:
            ExtractionResult with all extracted data.
        """
        self._logger.info(
            "Starting full extraction",
            extra={"text_length": len(text)},
        )

        # Extract locations
        try:
            locations = self.extract_locations(text)
        except ExtractionError:
            locations = []

        # Validate cities
        cities = self.extract_and_validate_cities(text) if locations else []

        # Extract dates
        try:
            dates_raw = self.extract_dates(text)
        except ExtractionError:
            dates_raw = []

        # Normalize dates if requested
        dates_normalized = None
        if normalize_dates and dates_raw:
            try:
                from ..dates import normalize_dates_fr

                dates_normalized = tuple(normalize_dates_fr(list(dates_raw)))
            except Exception as e:
                self._logger.warning(
                    "Date normalization failed",
                    extra={"error": str(e)},
                )

        result = ExtractionResult(
            locations=tuple(locations),
            cities=tuple(cities),
            dates_raw=tuple(dates_raw),
            dates_normalized=dates_normalized,
        )

        self._logger.info(
            "Extraction complete",
            extra={
                "locations": len(result.locations),
                "cities": len(result.cities),
                "dates": len(result.dates_raw),
            },
        )

        return result
