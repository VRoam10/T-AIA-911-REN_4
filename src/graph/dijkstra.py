"""Shortest-path computation using Dijkstra's algorithm.

This module declares the interface for computing the shortest path
between two stations in the transportation graph using Dijkstra's
algorithm.
"""

import heapq
from typing import Dict, List, Tuple

from .load_graph import Graph


def dijkstra(graph: Graph, start: str, end: str) -> Tuple[List[str], float]:
    """Compute the shortest path between two stations using Dijkstra.

    Parameters
    ----------
    graph:
        Transportation graph as produced by ``load_graph``.
    start:
        Identifier of the departure station.
    end:
        Identifier of the arrival station.

    Returns
    -------
    list[str], float
        The sequence of station identifiers representing the path from
        ``start`` to ``end`` (inclusive) and the total distance.
        If no path exists, returns ``([], float("inf"))``.
    """
    if start not in graph or end not in graph:
        return [], float("inf")

    distances: Dict[str, float] = {station: float("inf") for station in graph}
    previous: Dict[str, str] = {}
    distances[start] = 0.0

    heap: List[Tuple[float, str]] = [(0.0, start)]
    visited = set()

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
