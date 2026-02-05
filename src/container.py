"""Dependency injection container.

This module provides a simple DI container without external frameworks.
It allows registering and resolving dependencies for the application.

Design principles:
1. No magic - explicit registration and resolution
2. Testable - easy to swap implementations
3. Lazy loading - adapters instantiated on first use
4. Thread-safe - for web server contexts
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from .config import AppConfig, get_config

T = TypeVar("T")


@dataclass
class Container:
    """Dependency injection container.

    Usage:
        # Production
        container = Container.create_default()
        resolver = container.resolve(TravelResolverService)

        # Testing
        container = Container()
        container.register(StationExtractorPort, lambda: MockExtractor())
        extractor = container.resolve(StationExtractorPort)

    Attributes:
        config: Application configuration
    """

    config: AppConfig = field(default_factory=get_config)

    _factories: Dict[type[Any], Callable[[], Any]] = field(
        default_factory=dict, repr=False
    )
    _singletons: Dict[type[Any], Any] = field(default_factory=dict, repr=False)
    _singleton_types: set[type[Any]] = field(default_factory=set, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def register(
        self,
        port_type: type[Any],
        factory: Callable[[], Any],
        singleton: bool = True,
    ) -> None:
        """Register a factory for a port type.

        Args:
            port_type: The type (usually a Protocol) to register.
            factory: A callable that creates instances of the type.
            singleton: If True, only one instance is created.
        """
        with self._lock:
            self._factories[port_type] = factory
            if singleton:
                self._singleton_types.add(port_type)

    def resolve(self, port_type: type[Any]) -> Any:
        """Resolve an instance of a port type.

        Args:
            port_type: The type to resolve.

        Returns:
            An instance of the requested type.

        Raises:
            KeyError: If the type is not registered.
        """
        with self._lock:
            if port_type not in self._factories:
                raise KeyError(f"Type not registered: {port_type}")

            if port_type in self._singleton_types:
                if port_type not in self._singletons:
                    self._singletons[port_type] = self._factories[port_type]()
                return self._singletons[port_type]

            return self._factories[port_type]()

    def is_registered(self, port_type: type[Any]) -> bool:
        """Check if a type is registered.

        Args:
            port_type: The type to check.

        Returns:
            True if the type is registered.
        """
        return port_type in self._factories

    def clear_singletons(self) -> None:
        """Clear all cached singletons.

        Call this in tests to ensure fresh instances.
        """
        with self._lock:
            self._singletons.clear()

    def clear_all(self) -> None:
        """Clear all registrations and singletons.

        Call this to completely reset the container.
        """
        with self._lock:
            self._factories.clear()
            self._singletons.clear()
            self._singleton_types.clear()

    @classmethod
    def create_default(cls, config: Optional[AppConfig] = None) -> Container:
        """Create a container with default production bindings.

        This creates a fully configured container with all adapters
        registered and ready to use.

        Args:
            config: Optional configuration override.

        Returns:
            A configured Container instance.
        """
        from .adapters.cache import InMemoryCache
        from .adapters.geocoding import NominatimGeocoderAdapter
        from .adapters.graph import CSVGraphRepository, DijkstraRouteSolver
        from .adapters.nlp import RuleBasedIntentClassifier, RuleBasedStationExtractor
        from .adapters.rendering import FoliumMapRenderer
        from .ports.cache import CachePort
        from .ports.geocoding import GeocoderPort
        from .ports.graph import GraphRepositoryPort, RouteSolverPort
        from .ports.nlp import IntentClassifierPort, StationExtractorPort
        from .ports.rendering import MapRendererPort
        from .services import TravelResolverService

        config = config or get_config()
        container = cls(config=config)

        # Cache (shared across adapters)
        cache: InMemoryCache[Any] = InMemoryCache(name="global")
        container.register(CachePort, lambda: cache)

        # NLP
        container.register(
            IntentClassifierPort,
            lambda: RuleBasedIntentClassifier(),
        )

        # Station extractor based on config
        def create_station_extractor() -> StationExtractorPort:
            strategy = config.nlp.default_strategy
            if strategy == "rule_based":
                return RuleBasedStationExtractor(config.graph)
            elif strategy == "spacy":
                from .adapters.nlp import SpaCyNERAdapter

                return SpaCyNERAdapter(config.nlp, cache)
            elif strategy == "hf_ner":
                from .adapters.nlp import HuggingFaceNERAdapter

                return HuggingFaceNERAdapter(config.nlp, cache)
            else:
                return RuleBasedStationExtractor(config.graph)

        container.register(StationExtractorPort, create_station_extractor)

        # Geocoding
        container.register(
            GeocoderPort,
            lambda: NominatimGeocoderAdapter(config.geocoding, cache),
        )

        # Graph
        container.register(
            GraphRepositoryPort,
            lambda: CSVGraphRepository(config.graph),
        )
        container.register(
            RouteSolverPort,
            lambda: DijkstraRouteSolver(),
        )

        # Rendering
        container.register(
            MapRendererPort,
            lambda: FoliumMapRenderer(),
        )

        # Main service
        def create_travel_resolver() -> TravelResolverService:
            return TravelResolverService(
                intent_classifier=container.resolve(IntentClassifierPort),
                station_extractor=container.resolve(StationExtractorPort),
                graph_repository=container.resolve(GraphRepositoryPort),
                route_solver=container.resolve(RouteSolverPort),
                map_renderer=container.resolve(MapRendererPort),
            )

        container.register(TravelResolverService, create_travel_resolver)

        return container


# Global default container (lazy initialized)
_default_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get the default application container.

    Returns:
        The default Container instance (creates one if needed).
    """
    global _default_container
    if _default_container is None:
        with _container_lock:
            if _default_container is None:
                _default_container = Container.create_default()
    return _default_container


def reset_container() -> None:
    """Reset the default container.

    Call this in tests to ensure a fresh container.
    """
    global _default_container
    with _container_lock:
        if _default_container is not None:
            _default_container.clear_all()
        _default_container = None
