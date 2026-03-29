# Dijkstra vs A* (v2) — Algorithm Comparison

**Module**: `src/graph/`
**Scope**: Shortest-path computation over the French railway transportation graph

---

## Table of Contents

1. [Overview](#1-overview)
2. [Graph Data Model](#2-graph-data-model)
3. [Dijkstra (v1)](#3-dijkstra-v1)
   - [How It Works](#31-how-it-works)
   - [Complexity](#32-complexity)
   - [Implementation Details](#33-implementation-details)
4. [A* with Haversine Heuristic (v2)](#4-a-with-haversine-heuristic-v2)
   - [Motivation](#41-motivation)
   - [How It Works](#42-how-it-works)
   - [The Haversine Heuristic](#43-the-haversine-heuristic)
   - [Admissibility](#44-admissibility)
   - [Complexity](#45-complexity)
   - [Implementation Details](#46-implementation-details)
5. [Side-by-Side Comparison](#5-side-by-side-comparison)
6. [API Reference](#6-api-reference)
   - [dijkstra()](#61-dijkstra)
   - [astar()](#62-astar)
   - [benchmark_both()](#63-benchmark_both)
   - [DijkstraRouteSolver (adapter)](#64-dijkstraroutesolver-adapter)
7. [Usage Examples](#7-usage-examples)
8. [Benchmarking Guide](#8-benchmarking-guide)
9. [Architecture Notes](#9-architecture-notes)

---

## 1. Overview

This project uses two shortest-path algorithms to compute optimal railway routes:

| Version | File | Algorithm | Heuristic |
|---------|------|-----------|-----------|
| **v1** | `src/graph/dijkstra.py` | Dijkstra | None (blind search) |
| **v2** | `src/graph/dijkstrav2.py` | A* | Haversine great-circle distance |

Both algorithms are guaranteed to return the **same optimal path and distance** given the same graph. The difference lies in **how many nodes are explored** before reaching the destination — A* uses geographical coordinates to guide the search toward the goal, visiting fewer nodes on average.

The v1 implementation is also wrapped as a hexagonal architecture adapter in `src/adapters/graph/dijkstra_solver.py`, which adds domain model output, structured error handling, and logging.

---

## 2. Graph Data Model

Both algorithms operate on the same in-memory graph structure built by `src/graph/load_graph.py`.

```
Graph = Dict[str, List[Tuple[str, float]]]
        ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        node   list of (neighbour, distance_km)
```

The graph is loaded from two CSV files:

| File | Columns | Role |
|------|---------|------|
| `data/stations.csv` | `station_id`, `name`, `city`, `lat`, `lon` | Declares all nodes |
| `data/edges.csv` | `from_station_id`, `to_station_id`, `distance_km` | Declares directed edges |

The graph is **directed** — each row in `edges.csv` represents a one-way connection. Bidirectional connections appear as two separate rows (one per direction).

**Dataset size (current):** ~500 stations, ~4 874 edges.

```python
from src.graph.load_graph import load_graph

graph = load_graph("data/stations.csv", "data/edges.csv")
# graph["FR_RENNES"] → [("FR_COMBREE", 71.569), ("FR_BRETEIL", 20.618), ...]
```

---

## 3. Dijkstra (v1)

### 3.1 How It Works

Dijkstra's algorithm explores the graph **by increasing cost from the source**. It maintains a priority queue (min-heap) of `(cost, node)` pairs and a `distances` dictionary tracking the best-known cost to reach each node.

**Step-by-step:**

1. Initialize: `distances[start] = 0`, all others `= ∞`.
2. Push `(0.0, start)` onto the heap.
3. Pop the node `u` with the lowest cost.
4. If `u` was already visited, skip it.
5. If `u == end`, stop — optimal path found.
6. For each neighbour `v` of `u`: if `distances[u] + weight(u→v) < distances[v]`, update `distances[v]` and push `(new_distance, v)` onto the heap.
7. Reconstruct the path by backtracking through the `previous` dictionary.

```
Priority queue state (example, 3-node graph A→B→C):

Initial:  [(0, A)]
After A:  [(3, B), (10, C)]   ← explores B first (cheaper)
After B:  [(7, C), (10, C)]   ← updates C via B
After C:  done — path = [A, B, C], dist = 7
```

### 3.2 Complexity

| Metric | Value | Notes |
|--------|-------|-------|
| Time | O((V + E) log V) | V = nodes, E = edges; dominated by heap ops |
| Space | O(V) | `distances` + `previous` dictionaries + heap |
| Optimality | **Always optimal** | For non-negative edge weights |
| Completeness | **Always complete** | Finds a path if one exists |

### 3.3 Implementation Details

**File:** `src/graph/dijkstra.py`

```python
def dijkstra(graph: Graph, start: str, end: str) -> Tuple[List[str], float]:
```

Key implementation choices:

- **Early exit**: The loop breaks as soon as `end` is popped from the heap, avoiding unnecessary exploration of the rest of the graph.
- **Lazy deletion**: Stale heap entries (with outdated costs) are discarded with the `if u in visited: continue` guard — simpler than a decrease-key operation.
- **Path reconstruction**: The `previous` dictionary maps each node to its optimal predecessor. The path is built backwards from `end` to `start`, then reversed.
- **No-path sentinel**: Returns `([], float("inf"))` when `end` is unreachable.

---

## 4. A\* with Haversine Heuristic (v2)

### 4.1 Motivation

Dijkstra explores nodes uniformly in all directions from the source. On a geographical graph like a rail network, this means exploring stations to the west even when the destination is to the east. A* addresses this by adding a **heuristic estimate** of the remaining distance to the goal, biasing the search toward the destination.

### 4.2 How It Works

A* extends Dijkstra by scoring each node with:

```
f(n) = g(n) + h(n)
       ^^^^   ^^^^
       actual  heuristic estimate
       cost    of cost from n to goal
       from
       start
```

The heap stores `(f_score, g_score, node)` triples. Nodes with a lower estimated total cost are explored first.

**Step-by-step:**

1. Initialize: `g_scores[start] = 0`.
2. Push `(h(start), 0.0, start)` onto the heap.
3. Pop the node `u` with the lowest `f = g + h`.
4. If `u` was already visited, skip it.
5. If `u == end`, stop — optimal path found.
6. For each neighbour `v` of `u`: if `g[u] + weight(u→v) < g_scores[v]`, update `g_scores[v]`, record `previous[v] = u`, and push `(new_g + h(v), new_g, v)`.
7. Reconstruct the path identically to Dijkstra.

### 4.3 The Haversine Heuristic

The heuristic `h(n)` is the **great-circle (straight-line) distance** between node `n` and the destination, computed using the Haversine formula:

```python
def _haversine(lat1, lon1, lat2, lon2) -> float:  # returns km
    R = 6371.0  # Earth radius in km
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)² + cos(φ1)·cos(φ2)·sin(Δλ/2)²
    return R · 2 · atan2(√a, √(1−a))
```

GPS coordinates are loaded from `data/stations.csv` (`lat`, `lon` columns) via `load_coords()`.

**Graceful degradation**: If coordinates are missing for either the current node or the destination, `h(n)` returns `0.0`, making A* behave exactly like Dijkstra for that node.

### 4.4 Admissibility

A heuristic is **admissible** if it never overestimates the true cost. This is a required property for A* to guarantee an optimal solution.

The Haversine distance is admissible for a rail network because:
- It computes the straight-line distance between two GPS points on Earth.
- Rail tracks are always longer than the straight-line distance (they follow curves, terrain, and existing infrastructure).
- Therefore: `h(n) ≤ actual_rail_distance(n, goal)` always holds.

**Consequence**: A* with this heuristic **always returns the same optimal path as Dijkstra**, while exploring fewer nodes.

### 4.5 Complexity

| Metric | Value | Notes |
|--------|-------|-------|
| Time | O((V + E) log V) worst case | Same asymptotic bound as Dijkstra |
| Time (typical) | Significantly fewer nodes visited | Guided by heuristic |
| Space | O(V) | Same structures as Dijkstra |
| Optimality | **Always optimal** | Heuristic is admissible |
| Completeness | **Always complete** | Same guarantee as Dijkstra |

The improvement is in the **constant factor**: A* typically visits far fewer nodes before reaching the goal, leading to lower wall-clock time in practice.

### 4.6 Implementation Details

**File:** `src/graph/dijkstrav2.py`

The module exposes both a public API and internal `_core` functions for benchmarking:

```python
def astar(graph, start, end, coords) -> Tuple[List[str], float]
def _astar_core(graph, start, end, coords) -> Tuple[List[str], float, int]
def _dijkstra_core(graph, start, end) -> Tuple[List[str], float, int]
```

The `_core` variants additionally return `nodes_visited` (an integer counter incremented each time a node is finalized), used by `benchmark_both()`.

Key difference from v1: the heap tuple is `(f_score, g_score, node)` instead of `(distance, node)`. The `f_score` drives priority ordering; `g_score` is carried along to avoid redundant dictionary lookups.

---

## 5. Side-by-Side Comparison

### Algorithm Behaviour

| Property | Dijkstra (v1) | A* (v2) |
|----------|--------------|---------|
| Search strategy | Uniform-cost (BFS by cost) | Best-first (cost + heuristic) |
| Heuristic | None (`h = 0`) | Haversine great-circle distance |
| Nodes explored | All reachable nodes up to goal cost | Only promising nodes toward goal |
| Optimal path | Always | Always (heuristic is admissible) |
| Same result? | — | Yes, identical path and distance |
| Requires coordinates | No | Yes (falls back to Dijkstra if missing) |

### Code Structure

| Aspect | Dijkstra (v1) | A* (v2) |
|--------|--------------|---------|
| File | `src/graph/dijkstra.py` | `src/graph/dijkstrav2.py` |
| Heap entry | `(g, node)` | `(f, g, node)` |
| Priority key | `g(n)` — actual cost | `f(n) = g(n) + h(n)` |
| Extra input | None | `coords: Dict[str, Tuple[float, float]]` |
| Benchmark support | No | Yes — `benchmark_both()`, `AlgoStats` |

### When to Use Each

| Scenario | Recommended |
|----------|-------------|
| GPS coordinates unavailable | Dijkstra (v1) |
| Production routing (speed matters) | A* (v2) |
| Benchmarking and comparison | `benchmark_both()` in v2 |
| Hexagonal architecture / domain layer | `DijkstraRouteSolver` adapter |
| Quick integration test with minimal setup | Dijkstra (v1) |

---

## 6. API Reference

### 6.1 `dijkstra()`

**File:** `src/graph/dijkstra.py`

```python
def dijkstra(
    graph: Graph,
    start: str,
    end: str,
) -> Tuple[List[str], float]:
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `graph` | `Graph` | Transportation graph from `load_graph()` |
| `start` | `str` | Station identifier for the departure |
| `end` | `str` | Station identifier for the arrival |

**Returns:** `(path, distance_km)`

| Field | Type | Description |
|-------|------|-------------|
| `path` | `List[str]` | Ordered list of station IDs from start to end (inclusive). Empty if no path. |
| `distance_km` | `float` | Total distance in kilometres. `float("inf")` if no path. |

**Edge cases:**
- If `start` or `end` is not in the graph, returns `([], float("inf"))`.
- If `start == end`, returns `([start], 0.0)`.

---

### 6.2 `astar()`

**File:** `src/graph/dijkstrav2.py`

```python
def astar(
    graph: Graph,
    start: str,
    end: str,
    coords: Coordinates,
) -> Tuple[List[str], float]:
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `graph` | `Graph` | Transportation graph from `load_graph()` |
| `start` | `str` | Station identifier for the departure |
| `end` | `str` | Station identifier for the arrival |
| `coords` | `Dict[str, Tuple[float, float]]` | GPS coordinates `station_id → (lat, lon)` from `load_coords()` |

**Returns:** Identical signature to `dijkstra()` — `(path, distance_km)`.

Drop-in replacement: `astar()` can replace `dijkstra()` wherever `coords` are available.

---

### 6.3 `benchmark_both()`

**File:** `src/graph/dijkstrav2.py`

```python
def benchmark_both(
    graph: Graph,
    start: str,
    end: str,
    coords: Coordinates,
    *,
    runs: int = 10,
) -> Tuple[AlgoStats, AlgoStats]:
```

Runs both algorithms `runs` times and returns averaged performance statistics.

**Returns:** `(dijkstra_stats, astar_stats)` — a pair of `AlgoStats` dataclasses.

**`AlgoStats` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | `str` | `"Dijkstra"` or `"A*"` |
| `path` | `List[str]` | The computed path (from the last run) |
| `distance_km` | `float` | Total distance in km |
| `nodes_visited` | `int` | Number of nodes finalized (from the last run) |
| `time_ms` | `float` | Average wall-clock time across all runs, in milliseconds |
| `found` | `bool` | Property — `True` if `path` is non-empty |

---

### 6.4 `DijkstraRouteSolver` (adapter)

**File:** `src/adapters/graph/dijkstra_solver.py`

The hexagonal architecture adapter wrapping Dijkstra for use in the service layer.

```python
@dataclass
class DijkstraRouteSolver:
    def solve(
        self,
        graph: Graph,
        departure: str,
        arrival: str,
        stations: Optional[Dict[str, Station]] = None,
    ) -> RouteResult: ...

    def solve_safe(
        self,
        graph: Graph,
        departure: str,
        arrival: str,
        stations: Optional[Dict[str, Station]] = None,
    ) -> RouteResult: ...
```

| Method | Raises on failure | Returns on failure |
|--------|-------------------|--------------------|
| `solve()` | `StationNotFoundError`, `NoRouteFoundError` | — |
| `solve_safe()` | Never | `RouteResult(path=(), total_distance_km=inf)` |

**`RouteResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `path` | `Tuple[str, ...]` | Ordered station IDs |
| `total_distance_km` | `float` | Total route distance |
| `stations` | `Tuple[Station, ...]` | Resolved `Station` objects (if `stations` dict provided) |

---

## 7. Usage Examples

### Basic Dijkstra

```python
from src.graph.load_graph import load_graph
from src.graph.dijkstra import dijkstra

graph = load_graph("data/stations.csv", "data/edges.csv")
path, distance = dijkstra(graph, "FR_RENNES", "FR_BORDEAUX_ST_JEAN")

print(f"Distance: {distance:.1f} km")
print(f"Path: {' → '.join(path)}")
```

### A* (drop-in replacement)

```python
from src.graph.load_graph import load_graph
from src.graph.dijkstrav2 import astar, load_coords

graph  = load_graph("data/stations.csv", "data/edges.csv")
coords = load_coords("data/stations.csv")

path, distance = astar(graph, "FR_RENNES", "FR_BORDEAUX_ST_JEAN", coords)

print(f"Distance: {distance:.1f} km")
print(f"Path ({len(path)} stops): {' → '.join(path)}")
```

### Benchmarking

```python
from src.graph.load_graph import load_graph
from src.graph.dijkstrav2 import benchmark_both, load_coords

graph  = load_graph("data/stations.csv", "data/edges.csv")
coords = load_coords("data/stations.csv")

dijk, astar_stats = benchmark_both(
    graph, "FR_RENNES", "FR_BORDEAUX_ST_JEAN", coords, runs=20
)

print(f"Dijkstra — {dijk.time_ms:.3f} ms avg, {dijk.nodes_visited} nodes visited")
print(f"A*       — {astar_stats.time_ms:.3f} ms avg, {astar_stats.nodes_visited} nodes visited")
print(f"Speedup  — {dijk.time_ms / astar_stats.time_ms:.2f}x")
print(f"Node reduction — {(1 - astar_stats.nodes_visited / dijk.nodes_visited) * 100:.1f}%")
assert dijk.distance_km == astar_stats.distance_km, "Both must find the same optimal cost"
```

### Using the hexagonal adapter

```python
from src.container import Container
from src.services import TravelResolverService

container = Container.create_default()
service   = container.resolve(TravelResolverService)

result, error = service.resolve_safe("Je veux aller de Rennes à Bordeaux")
if not error:
    print(f"Path: {' → '.join(result.path)}")
    print(f"Distance: {result.total_distance_km:.1f} km")
```

---

## 8. Benchmarking Guide

### Running the benchmark

```bash
python - <<'EOF'
from src.graph.load_graph import load_graph
from src.graph.dijkstrav2 import benchmark_both, load_coords

graph  = load_graph("data/stations.csv", "data/edges.csv")
coords = load_coords("data/stations.csv")

pairs = [
    ("FR_RENNES",        "FR_BORDEAUX_ST_JEAN"),
    ("FR_LILLE_FLANDRES", "FR_MARSEILLE_ST_CHARLES"),
    ("FR_STRASBOURG",    "FR_BREST"),
]

for start, end in pairs:
    d, a = benchmark_both(graph, start, end, coords, runs=20)
    reduction = (1 - a.nodes_visited / d.nodes_visited) * 100 if d.nodes_visited else 0
    print(f"{start.split('_',1)[1]:25} → {end.split('_',1)[1]:25} | "
          f"Dijkstra {d.time_ms:6.3f}ms ({d.nodes_visited:4} nodes) | "
          f"A* {a.time_ms:6.3f}ms ({a.nodes_visited:4} nodes) | "
          f"−{reduction:.0f}% nodes")
EOF
```

### What to expect

A* consistently visits **fewer nodes** than Dijkstra for geographically distant station pairs because the haversine heuristic directs the search toward the destination. The gain is proportional to the spatial spread of the graph relative to the route length:

- **Short routes** (nearby stations): small improvement, heuristic has little room to prune.
- **Long cross-country routes**: larger improvement, many irrelevant branches pruned early.

The wall-clock speedup may be modest on Python due to heap operation overhead, but `nodes_visited` is the accurate measure of algorithmic improvement.

### Interpreting `nodes_visited`

`nodes_visited` is incremented each time a node is popped from the heap and **not already in `visited`** — i.e., each time a node is permanently finalized. It is the standard measure for comparing graph search algorithms independent of hardware.

---

## 9. Architecture Notes

### File Map

```
src/graph/
├── __init__.py               # Package docstring
├── load_graph.py             # Graph type definition + CSV loader
├── dijkstra.py               # v1: pure Dijkstra, no dependencies
├── dijkstrav2.py             # v2: A* + Dijkstra cores + benchmark utility
└── documentation/
    └── dijkstra_vs_astar.md  # This document

src/adapters/graph/
├── __init__.py
├── csv_repository.py         # CSVGraphRepository — loads and caches graph
└── dijkstra_solver.py        # DijkstraRouteSolver — hexagonal adapter over v1
```

### Design Decisions

**Why keep v1 alongside v2?**
`dijkstra.py` remains the canonical, dependency-free implementation. It is used directly by the adapter layer (`dijkstra_solver.py`) and by unit tests. `dijkstrav2.py` is a separate module focused on the A* improvement and benchmarking tooling — it does not replace v1.

**Why is the adapter based on v1 and not v2?**
The `DijkstraRouteSolver` adapter was written against the existing v1 interface. Migrating it to use `astar()` would require passing GPS coordinates into the adapter, which would change the `RouteSolverPort` interface. This is a future improvement that can be made without altering the public API by injecting `coords` at construction time.

**Why a directed graph?**
`edges.csv` contains directional rail segments. Bidirectional connections are represented as two rows. This allows the dataset to model one-way segments (maintenance lines, etc.) without special cases in the algorithm.

**Heuristic consistency (monotonicity)**
In addition to admissibility, the Haversine heuristic is **consistent**: for any edge `(u, v)` with cost `w`, `h(u) ≤ w(u, v) + h(v)` (triangle inequality on geographic space). Consistency guarantees that A* never re-opens a closed node, making the `visited` set optimization valid.
