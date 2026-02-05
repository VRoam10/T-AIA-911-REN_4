"""Typed domain errors for the Travel Order Resolver.

These error types replace silent exception swallowing with explicit,
typed errors that can be handled appropriately at each layer.

All errors inherit from TravelResolverError and can optionally
wrap a root cause exception for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TravelResolverError(Exception):
    """Base error for the travel resolver domain.

    All domain-specific errors inherit from this class.

    Attributes:
        message: Human-readable error description
        cause: Optional underlying exception that caused this error
    """

    message: str
    cause: Optional[Exception] = field(default=None, repr=False)

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message}: {self.cause}"
        return self.message

    def __post_init__(self) -> None:
        super().__init__(self.message)


@dataclass
class ExtractionError(TravelResolverError):
    """Failed to extract entities from text.

    Replaces silent failures in:
    - strategies.py:189-193 (HF fallback)
    - strategies.py:207-211 (dates fallback)

    Attributes:
        strategy_used: Name of the extraction strategy that failed
        fallback_attempted: Whether a fallback strategy was tried
    """

    strategy_used: str = ""
    fallback_attempted: bool = False


@dataclass
class IntentClassificationError(TravelResolverError):
    """Failed to classify user intent.

    Attributes:
        detected_intent: The intent that was detected (if any)
    """

    detected_intent: Optional[str] = None


@dataclass
class GeocodingError(TravelResolverError):
    """Failed to geocode a location.

    Replaces silent None returns in strategies.py:92-98.

    Attributes:
        query: The location query that failed
        is_rate_limited: Whether the failure was due to rate limiting
    """

    query: str = ""
    is_rate_limited: bool = False


@dataclass
class GraphError(TravelResolverError):
    """Graph loading or data integrity error.

    Attributes:
        file_path: Path to the graph data file if relevant
    """

    file_path: Optional[str] = None


@dataclass
class NoRouteFoundError(TravelResolverError):
    """No path exists between the requested stations.

    Attributes:
        departure: Departure station code
        arrival: Arrival station code
    """

    departure: str = ""
    arrival: str = ""


@dataclass
class StationNotFoundError(TravelResolverError):
    """Station code not found in the graph.

    Attributes:
        station_code: The station code that was not found
    """

    station_code: str = ""


@dataclass
class ASRError(TravelResolverError):
    """ASR transcription failed.

    Attributes:
        model_id: The ASR model that failed
        device: The device the model was running on
        audio_path: Path to the audio file if relevant
    """

    model_id: str = ""
    device: str = ""
    audio_path: Optional[str] = None


@dataclass
class ConfigurationError(TravelResolverError):
    """Invalid or missing configuration.

    Attributes:
        setting_name: Name of the problematic setting
        expected_type: Expected type or format
    """

    setting_name: str = ""
    expected_type: Optional[str] = None


@dataclass
class RenderingError(TravelResolverError):
    """Map or visualization rendering failed.

    Replaces silent catch in pipeline.py:121-122.

    Attributes:
        output_path: Path where rendering was attempted
        renderer_type: Type of renderer that failed
    """

    output_path: Optional[str] = None
    renderer_type: str = ""
