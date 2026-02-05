"""Travel resolver service - Main orchestrator.

This service replaces the pipeline orchestration in pipeline.py
with proper dependency injection and error handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..domain.errors import (
    ExtractionError,
    IntentClassificationError,
    NoRouteFoundError,
    RenderingError,
)
from ..domain.models import Intent, RouteResult, StationExtractionResult
from ..ports.graph import GraphRepositoryPort, RouteSolverPort
from ..ports.nlp import IntentClassifierPort, StationExtractorPort
from ..ports.rendering import MapRendererPort


@dataclass
class TravelResolverService:
    """Main service for resolving travel orders.

    This service orchestrates the full pipeline:
    1. Intent classification
    2. Station extraction
    3. Route computation
    4. Optional map rendering

    Replaces: pipeline.py:48-125 (solve_travel_order)

    Attributes:
        intent_classifier: Classifies user intent
        station_extractor: Extracts stations from text
        graph_repository: Loads transportation graph
        route_solver: Computes shortest paths
        map_renderer: Optional map rendering
    """

    intent_classifier: IntentClassifierPort
    station_extractor: StationExtractorPort
    graph_repository: GraphRepositoryPort
    route_solver: RouteSolverPort
    map_renderer: Optional[MapRendererPort] = None

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def resolve(
        self,
        sentence: str,
        generate_map: bool = False,
        map_output_path: Optional[Path] = None,
    ) -> RouteResult:
        """Resolve a travel order from natural language.

        Args:
            sentence: The input sentence describing a travel request.
            generate_map: Whether to generate a map visualization.
            map_output_path: Path for the map file (required if generate_map=True).

        Returns:
            RouteResult with the computed route.

        Raises:
            IntentClassificationError: If intent is not a valid trip.
            ExtractionError: If station extraction fails.
            NoRouteFoundError: If no path exists between stations.
            RenderingError: If map generation fails.
        """
        self._logger.info(
            "Starting travel resolution",
            extra={"sentence_length": len(sentence)},
        )

        # Step 1: Intent classification
        intent = self.intent_classifier.classify(sentence)
        self._logger.info("Intent classified", extra={"intent": intent.name})

        if intent == Intent.UNKNOWN:
            raise IntentClassificationError(
                "Empty or invalid input",
                detected_intent="UNKNOWN",
            )
        elif intent == Intent.NOT_FRENCH:
            raise IntentClassificationError(
                "Input is not in French",
                detected_intent="NOT_FRENCH",
            )
        elif intent == Intent.NOT_TRIP:
            raise IntentClassificationError(
                "Input is not a travel request",
                detected_intent="NOT_TRIP",
            )

        # Step 2: Station extraction
        extraction = self.station_extractor.extract(sentence)
        self._logger.info(
            "Stations extracted",
            extra={
                "departure": extraction.departure,
                "arrival": extraction.arrival,
                "success": extraction.is_success,
            },
        )

        if not extraction.is_success:
            raise ExtractionError(
                extraction.error or "Could not extract both departure and arrival",
                strategy_used=type(self.station_extractor).__name__,
            )

        # Step 3: Graph loading
        graph = self.graph_repository.load()
        self._logger.debug("Graph loaded", extra={"nodes": len(graph)})

        # Step 4: Route computation
        route = self.route_solver.solve(
            graph,
            extraction.departure,  # type: ignore
            extraction.arrival,  # type: ignore
        )
        self._logger.info(
            "Route computed",
            extra={
                "stops": route.num_stops,
                "distance_km": route.total_distance_km,
            },
        )

        if route.is_empty:
            raise NoRouteFoundError(
                f"No path from {extraction.departure} to {extraction.arrival}",
                departure=extraction.departure or "",
                arrival=extraction.arrival or "",
            )

        # Step 5: Optional map generation
        if generate_map and map_output_path and self.map_renderer:
            try:
                # Resolve station details for map
                stations_list = [
                    self.graph_repository.get_station(code) for code in route.path
                ]
                stations = tuple(s for s in stations_list if s is not None)

                if stations:
                    self.map_renderer.render(stations, map_output_path)
                    self._logger.info(
                        "Map generated",
                        extra={"path": str(map_output_path)},
                    )
            except RenderingError:
                raise
            except Exception as e:
                # Log but don't fail the entire request
                self._logger.warning(
                    "Map generation failed",
                    extra={"error": str(e)},
                )

        return route

    def resolve_safe(
        self,
        sentence: str,
        generate_map: bool = False,
        map_output_path: Optional[Path] = None,
    ) -> tuple[Optional[RouteResult], Optional[str]]:
        """Resolve a travel order, returning error message instead of raising.

        This method provides backward compatibility with the old API
        that returned error strings.

        Args:
            sentence: The input sentence.
            generate_map: Whether to generate a map.
            map_output_path: Path for the map file.

        Returns:
            Tuple of (RouteResult or None, error message or None).
        """
        try:
            result = self.resolve(sentence, generate_map, map_output_path)
            return result, None
        except IntentClassificationError as e:
            return None, f"Error: {e.message}"
        except ExtractionError as e:
            return None, f"Extraction error: {e.message}"
        except NoRouteFoundError as e:
            return None, f"No path found between {e.departure} and {e.arrival}"
        except Exception as e:
            self._logger.exception("Unexpected error in travel resolution")
            return None, f"Error: {e}"

    def format_result(self, route: RouteResult, map_path: Optional[Path] = None) -> str:
        """Format route result as human-readable string.

        Provides backward compatibility with the old string-based API.

        Args:
            route: The computed route.
            map_path: Optional path to generated map.

        Returns:
            Formatted result string.
        """
        path_str = " -> ".join(route.path)
        result = (
            f"Shortest path: {path_str}\nTotal distance: {route.total_distance_km} km"
        )

        if map_path:
            result += f"\nMap saved to: {map_path}"

        return result
