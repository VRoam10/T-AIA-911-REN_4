"""A* shortest-path computation as an improvement over Dijkstra.

This module provides:
- ``astar()``: A* algorithm with haversine heuristic (drop-in replacement for dijkstra)
- ``load_coords()``: Load GPS coordinates from stations.csv
- ``benchmark_both()``: Run both algorithms and return averaged comparison stats
"""

import csv
import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .load_graph import Graph

# station_id -> (lat_degrees, lon_degrees)
Coordinates = Dict[str, Tuple[float, float]]


def load_coords(stations_path: str) -> Coordinates:
    """Load GPS coordinates from stations.csv.

    Parameters
    ----------
    stations_path:
        Path to the stations CSV file (must have columns: station_id, lat, lon).

    Returns
    -------
    dict
        Mapping station_id -> (lat, lon).
    """
    coords: Coordinates = {}
    with open(stations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                coords[row["station_id"]] = (float(row["lat"]), float(row["lon"]))
            except (KeyError, ValueError):
                pass
    return coords


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line (great-circle) distance in km between two GPS points.

    Used as the A* admissible heuristic: never overestimates real rail distance.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Internal cores (return nodes_visited count for benchmarking)
# ---------------------------------------------------------------------------

def _astar_core(
    graph: Graph,
    start: str,
    end: str,
    coords: Coordinates,
) -> Tuple[List[str], float, int]:
    """A* core implementation.

    Returns (path, distance_km, nodes_visited).
    Falls back to zero heuristic (= Dijkstra) if coords are missing.
    """
    if start not in graph or end not in graph:
        return [], float("inf"), 0

    end_coords = coords.get(end)

    def h(node: str) -> float:
        """Haversine heuristic to goal, 0 if coords unavailable."""
        if end_coords is None or node not in coords:
            return 0.0
        lat, lon = coords[node]
        return _haversine(lat, lon, end_coords[0], end_coords[1])

    g_scores: Dict[str, float] = {start: 0.0}
    previous: Dict[str, str] = {}
    # heap stores (f_score, g_score, node)
    heap: List[Tuple[float, float, str]] = [(h(start), 0.0, start)]
    visited: set = set()
    nodes_visited = 0

    while heap:
        _f, g, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)
        nodes_visited += 1

        if u == end:
            break

        for v, weight in graph.get(u, []):
            new_g = g + weight
            if new_g < g_scores.get(v, float("inf")):
                g_scores[v] = new_g
                previous[v] = u
                heapq.heappush(heap, (new_g + h(v), new_g, v))

    if g_scores.get(end, float("inf")) == float("inf"):
        return [], float("inf"), nodes_visited

    path: List[str] = []
    current = end
    while current in previous or current == start:
        path.append(current)
        if current == start:
            break
        current = previous[current]
    path.reverse()

    return path, g_scores[end], nodes_visited


def _dijkstra_core(
    graph: Graph,
    start: str,
    end: str,
) -> Tuple[List[str], float, int]:
    """Dijkstra core – mirrors graph/dijkstra.py but also returns nodes_visited."""
    if start not in graph or end not in graph:
        return [], float("inf"), 0

    distances: Dict[str, float] = {station: float("inf") for station in graph}
    previous: Dict[str, str] = {}
    distances[start] = 0.0

    heap: List[Tuple[float, str]] = [(0.0, start)]
    visited: set = set()
    nodes_visited = 0

    while heap:
        current_distance, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)
        nodes_visited += 1

        if u == end:
            break

        for v, weight in graph.get(u, []):
            new_distance = current_distance + weight
            if new_distance < distances.get(v, float("inf")):
                distances[v] = new_distance
                previous[v] = u
                heapq.heappush(heap, (new_distance, v))

    if distances[end] == float("inf"):
        return [], float("inf"), nodes_visited

    path: List[str] = []
    current = end
    while current in previous or current == start:
        path.append(current)
        if current == start:
            break
        current = previous[current]
    path.reverse()

    return path, distances[end], nodes_visited


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def astar(
    graph: Graph,
    start: str,
    end: str,
    coords: Coordinates,
) -> Tuple[List[str], float]:
    """Compute the shortest path using A* with haversine heuristic.

    Drop-in replacement for ``dijkstra()`` – same return signature.

    Parameters
    ----------
    graph:
        Transportation graph as produced by ``load_graph``.
    start:
        Identifier of the departure station.
    end:
        Identifier of the arrival station.
    coords:
        GPS coordinates indexed by station_id (from ``load_coords``).

    Returns
    -------
    list[str], float
        The sequence of station identifiers from start to end (inclusive)
        and the total distance in km.
        If no path exists, returns ``([], float("inf"))``.
    """
    path, distance, _ = _astar_core(graph, start, end, coords)
    return path, distance


@dataclass
class AlgoStats:
    """Performance statistics from a single pathfinding run."""

    algorithm: str
    path: List[str] = field(default_factory=list)
    distance_km: float = float("inf")
    nodes_visited: int = 0
    time_ms: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.path)


def benchmark_both(
    graph: Graph,
    start: str,
    end: str,
    coords: Coordinates,
    *,
    runs: int = 10,
) -> Tuple[AlgoStats, AlgoStats]:
    """Run Dijkstra and A* multiple times and return averaged statistics.

    Parameters
    ----------
    graph:
        Transportation graph.
    start, end:
        Station identifiers.
    coords:
        GPS coordinates for A* heuristic.
    runs:
        Number of repetitions used to average timing noise.

    Returns
    -------
    (dijkstra_stats, astar_stats)
        Both ``AlgoStats`` objects with averaged ``time_ms`` and last-run
        ``nodes_visited``.
    """
    # --- Dijkstra ---
    dijk_times: List[float] = []
    dijk_path, dijk_dist, dijk_nodes = [], float("inf"), 0
    for _ in range(runs):
        t0 = time.perf_counter()
        dijk_path, dijk_dist, dijk_nodes = _dijkstra_core(graph, start, end)
        dijk_times.append((time.perf_counter() - t0) * 1000)

    # --- A* ---
    astar_times: List[float] = []
    astar_path, astar_dist, astar_nodes = [], float("inf"), 0
    for _ in range(runs):
        t0 = time.perf_counter()
        astar_path, astar_dist, astar_nodes = _astar_core(graph, start, end, coords)
        astar_times.append((time.perf_counter() - t0) * 1000)

    return (
        AlgoStats("Dijkstra", dijk_path, dijk_dist, dijk_nodes, sum(dijk_times) / runs),
        AlgoStats("A*", astar_path, astar_dist, astar_nodes, sum(astar_times) / runs),
    )
