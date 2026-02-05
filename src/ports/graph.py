"""Graph ports - Abstractions for graph loading and routing.

These protocols define the contracts for graph operations, including
loading transportation networks and computing shortest paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, Tuple

if TYPE_CHECKING:
    from ..domain.models import RouteResult, Station

# Graph type alias - preserves compatibility with load_graph.py:10
# Maps station code -> list of (neighbor_code, distance_km)
Graph = Mapping[str, Sequence[Tuple[str, float]]]


class GraphRepositoryPort(Protocol):
    """Port for loading graph data.

    Replaces: graph/load_graph.py:13-33 (load_graph function)
    Implementation: adapters/graph/csv_repository.py

    The repository is responsible for loading and caching the
    transportation network graph from persistent storage.
    """

    def load(self) -> Graph:
        """Load the transportation graph.

        Returns:
            The graph as a mapping of station codes to neighbors.
        """
        ...

    def get_station(self, code: str) -> Optional[Station]:
        """Get station details by code.

        Args:
            code: The station code to look up (e.g., 'FR_RENNES').

        Returns:
            Station with full details, or None if not found.
        """
        ...

    def list_stations(self) -> Sequence[Station]:
        """List all stations in the graph.

        Returns:
            Sequence of all stations with their details.
        """
        ...


class RouteSolverPort(Protocol):
    """Port for route computation.

    Replaces: pipeline.py:34 (PathFinder type alias)
    Wraps: graph/dijkstra.py:14-73

    The solver computes optimal paths through the transportation network.
    """

    def solve(
        self,
        graph: Graph,
        departure: str,
        arrival: str,
    ) -> RouteResult:
        """Find the shortest path between two stations.

        Args:
            graph: The transportation network graph.
            departure: Departure station code.
            arrival: Arrival station code.

        Returns:
            RouteResult with path, distance, and station details.
        """
        ...
