"""Domain layer - Core business models and errors.

This module contains immutable domain models and typed errors
used throughout the application. No external dependencies.
"""

from .errors import (
    ASRError,
    ConfigurationError,
    ExtractionError,
    GeocodingError,
    GraphError,
    NoRouteFoundError,
    StationNotFoundError,
    TravelResolverError,
)
from .models import (
    City,
    ExtractionResult,
    GeoLocation,
    Intent,
    RouteResult,
    Station,
    StationExtractionResult,
    TranscriptionResult,
    TranscriptionSegment,
)

__all__ = [
    # Models
    "GeoLocation",
    "Station",
    "StationExtractionResult",
    "RouteResult",
    "TranscriptionSegment",
    "TranscriptionResult",
    "City",
    "Intent",
    "ExtractionResult",
    # Errors
    "TravelResolverError",
    "ExtractionError",
    "GeocodingError",
    "GraphError",
    "NoRouteFoundError",
    "StationNotFoundError",
    "ASRError",
    "ConfigurationError",
]
