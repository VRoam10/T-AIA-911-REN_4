"""Dijkstra Route Solver adapter.

This adapter wraps the existing Dijkstra implementation and adds:
- Domain model output (RouteResult)
- Station resolution
- Better error handling
- Logging
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ...domain.errors import NoRouteFoundError, StationNotFoundError
from ...domain.models import RouteResult, Station
from ...ports.graph import Graph


@dataclass
class DijkstraRouteSolver:
    """Route solver using Dijkstra's shortest path algorithm.

    This adapter implements RouteSolverPort and wraps the logic
    from graph/dijkstra.py with additional features.
    """

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def solve(
        self,
        graph: Graph,
        departure: str,
        arrival: str,
        stations: Optional[Dict[str, Station]] = None,
    ) -> RouteResult:
        """Find the shortest path between two stations.

        Args:
            graph: The transportation network graph.
            departure: Departure station code.
            arrival: Arrival station code.
            stations: Optional station metadata for resolving details.

        Returns:
            RouteResult with path, distance, and station details.

        Raises:
            StationNotFoundError: If departure or arrival not in graph.
            NoRouteFoundError: If no path exists.
        """
        self._logger.debug(
            "Solving route",
            extra={"departure": departure, "arrival": arrival},
        )

        # Validate inputs
        if departure not in graph:
            raise StationNotFoundError(
                f"Departure station not in graph: {departure}",
                station_code=departure,
            )
        if arrival not in graph:
            raise StationNotFoundError(
                f"Arrival station not in graph: {arrival}",
                station_code=arrival,
            )

        # Run Dijkstra
        path, distance = self._dijkstra(graph, departure, arrival)

        if not path:
            self._logger.warning(
                "No route found",
                extra={"departure": departure, "arrival": arrival},
            )
            raise NoRouteFoundError(
                f"No path from {departure} to {arrival}",
                departure=departure,
                arrival=arrival,
            )

        # Resolve station details if available
        resolved_stations: Tuple[Station, ...] = ()
        if stations:
            resolved_stations = tuple(
                stations[code] for code in path if code in stations
            )

        self._logger.info(
            "Route found",
            extra={
                "departure": departure,
                "arrival": arrival,
                "stops": len(path),
                "distance_km": distance,
            },
        )

        return RouteResult(
            path=tuple(path),
            total_distance_km=distance,
            stations=resolved_stations,
        )

    def solve_safe(
        self,
        graph: Graph,
        departure: str,
        arrival: str,
        stations: Optional[Dict[str, Station]] = None,
    ) -> RouteResult:
        """Find the shortest path, returning empty result on failure.

        Like solve(), but returns an empty RouteResult instead of raising
        exceptions. Useful for backward compatibility with existing code.

        Args:
            graph: The transportation network graph.
            departure: Departure station code.
            arrival: Arrival station code.
            stations: Optional station metadata for resolving details.

        Returns:
            RouteResult with path and distance, or empty result.
        """
        if departure not in graph or arrival not in graph:
            return RouteResult(path=(), total_distance_km=float("inf"))

        path, distance = self._dijkstra(graph, departure, arrival)

        if not path:
            return RouteResult(path=(), total_distance_km=float("inf"))

        resolved_stations: Tuple[Station, ...] = ()
        if stations:
            resolved_stations = tuple(
                stations[code] for code in path if code in stations
            )

        return RouteResult(
            path=tuple(path),
            total_distance_km=distance,
            stations=resolved_stations,
        )

    def _dijkstra(
        self, graph: Graph, start: str, end: str
    ) -> Tuple[List[str], float]:
        """Core Dijkstra algorithm implementation.

        This is a direct port of graph/dijkstra.py:14-73.
        """
        if start not in graph or end not in graph:
            return [], float("inf")

        distances: Dict[str, float] = {station: float("inf") for station in graph}
        previous: Dict[str, str] = {}
        distances[start] = 0.0

        heap: List[Tuple[float, str]] = [(0.0, start)]
        visited: set[str] = set()

        while heap:
            current_distance, u = heapq.heappop(heap)

            if u in visited:
                continue

            visited.add(u)

            if u == end:
                break

            for v, weight in graph.get(u, []):
                new_distance = current_distance + weight
                if new_distance < distances.get(v, float("inf")):
                    distances[v] = new_distance
                    previous[v] = u
                    heapq.heappush(heap, (new_distance, v))

        if distances[end] == float("inf"):
            return [], float("inf")

        path: List[str] = []
        current = end
        while current in previous or current == start:
            path.append(current)
            if current == start:
                break
            current = previous[current]

        path.reverse()
        return path, distances[end]
