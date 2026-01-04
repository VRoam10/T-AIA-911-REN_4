"""Shortest-path computation using Dijkstra's algorithm.

This module declares the interface for computing the shortest path
between two stations in the transportation graph using Dijkstra's
algorithm.
"""

from typing import List, Tuple

from .load_graph import Graph


def dijkstra(graph: Graph, start: str, end: str) -> Tuple[List[str], float]:
    """Compute the shortest path between two stations.

    Parameters
    ----------
    graph:
        The transportation graph as produced by ``load_graph``.
    start:
        The identifier of the departure station.
    end:
        The identifier of the arrival station.

    Returns
    -------
    list[str], float
        A tuple containing the sequence of station identifiers
        representing the path, and the total cost (for example
        distance or duration) associated with that path.

    Notes
    -----
    Dijkstra's algorithm will be implemented manually in this module,
    and its time and space complexity will be analyzed as part of the
    project. For now, the function is left unimplemented to keep the
    focus on defining a clean, testable interface.
    """
    raise NotImplementedError("Dijkstra's algorithm is not implemented yet.")

