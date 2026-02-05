"""Immutable domain models for the Travel Order Resolver.

All models are frozen dataclasses with slots for memory efficiency.
These models have no external dependencies and represent the core
business concepts of the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Intent(Enum):
    """User intent classification.

    Represents the detected intent of a user's input sentence.
    Used to determine whether to proceed with travel resolution.
    """

    TRIP = auto()
    NOT_TRIP = auto()
    NOT_FRENCH = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class GeoLocation:
    """GPS coordinates representing a geographic location."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        """Validate coordinate ranges."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(
                f"Latitude must be between -90 and 90, got {self.latitude}"
            )
        if not -180 <= self.longitude <= 180:
            raise ValueError(
                f"Longitude must be between -180 and 180, got {self.longitude}"
            )


@dataclass(frozen=True, slots=True)
class Station:
    """A train station with its location and metadata.

    Attributes:
        code: Unique station identifier (e.g., 'FR_RENNES')
        name: Human-readable station name
        city: City where the station is located
        location: GPS coordinates of the station
    """

    code: str
    name: str
    city: str
    location: GeoLocation


@dataclass(frozen=True, slots=True)
class StationExtractionResult:
    """Result of extracting stations from natural language.

    This immutable result type replaces the mutable dataclass
    in nlp/extract_stations.py:30-48.

    Attributes:
        departure: Station code for departure, if detected
        arrival: Station code for arrival, if detected
        raw_locations: Raw location strings extracted from text
        confidence: Confidence score of the extraction (0.0 to 1.0)
        error: Optional error message if extraction failed
    """

    departure: Optional[str] = None
    arrival: Optional[str] = None
    raw_locations: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if both departure and arrival were extracted."""
        return self.departure is not None and self.arrival is not None

    @property
    def is_success(self) -> bool:
        """Check if extraction succeeded without errors."""
        return self.error is None and self.is_complete


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Result of route computation between stations.

    Attributes:
        path: Ordered tuple of station codes forming the route
        total_distance_km: Total distance of the route in kilometers
        stations: Resolved station details for each stop
    """

    path: tuple[str, ...]
    total_distance_km: float
    stations: tuple[Station, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        """Check if no route was found."""
        return len(self.path) == 0

    @property
    def num_stops(self) -> int:
        """Return the number of stops in the route."""
        return len(self.path)


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """A segment of transcribed audio with timestamps.

    Attributes:
        start_seconds: Start time of the segment
        end_seconds: End time of the segment
        text: Transcribed text for this segment
    """

    start_seconds: float
    end_seconds: float
    text: str

    @property
    def duration_seconds(self) -> float:
        """Return the duration of this segment."""
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Result of ASR transcription.

    Attributes:
        full_text: Complete transcribed text
        segments: Individual transcription segments with timestamps
        language: Detected language code
        language_probability: Confidence in language detection
        duration_seconds: Total audio duration if available
    """

    full_text: str
    segments: tuple[TranscriptionSegment, ...] = field(default_factory=tuple)
    language: str = "fr"
    language_probability: float = 1.0
    duration_seconds: Optional[float] = None


@dataclass(frozen=True, slots=True)
class City:
    """A validated city with geocoding information.

    This immutable model replaces the CityDict TypedDict
    in strategies.py:17-25.

    Attributes:
        name: City name
        location: GPS coordinates
        country: Country code or name
        admin_area: Administrative area (state, region, etc.)
    """

    name: str
    location: GeoLocation
    country: str = ""
    admin_area: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Complete extraction result from NLP processing.

    This immutable model replaces the ExtractionResult TypedDict
    in strategies.py:35-43.

    Attributes:
        locations: Raw location strings extracted
        cities: Validated cities with coordinates
        dates_raw: Raw date strings as found in text
        dates_normalized: ISO-formatted dates if normalization enabled
    """

    locations: tuple[str, ...] = field(default_factory=tuple)
    cities: tuple[City, ...] = field(default_factory=tuple)
    dates_raw: tuple[str, ...] = field(default_factory=tuple)
    dates_normalized: Optional[tuple[str, ...]] = None

    @property
    def has_locations(self) -> bool:
        """Check if any locations were extracted."""
        return len(self.locations) > 0

    @property
    def has_validated_cities(self) -> bool:
        """Check if any cities were validated via geocoding."""
        return len(self.cities) > 0

    @property
    def has_dates(self) -> bool:
        """Check if any dates were extracted."""
        return len(self.dates_raw) > 0
