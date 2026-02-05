"""Ports layer - Abstract interfaces (Protocols) for the application.

Ports define the contracts between the application core and external
adapters. They enable dependency injection and make the system testable.

This module follows the Hexagonal Architecture pattern:
- Input ports: How the application is driven (services)
- Output ports: How the application drives external systems (adapters)
"""

from .asr import ASRModelPort
from .cache import CachePort
from .geocoding import GeocoderPort
from .graph import Graph, GraphRepositoryPort, RouteSolverPort
from .nlp import (
    DateExtractorPort,
    IntentClassifierPort,
    LocationExtractorPort,
    StationExtractorPort,
)
from .rendering import MapRendererPort

__all__ = [
    # NLP
    "StationExtractorPort",
    "LocationExtractorPort",
    "DateExtractorPort",
    "IntentClassifierPort",
    # Graph
    "Graph",
    "GraphRepositoryPort",
    "RouteSolverPort",
    # Geocoding
    "GeocoderPort",
    # ASR
    "ASRModelPort",
    # Rendering
    "MapRendererPort",
    # Cache
    "CachePort",
]
