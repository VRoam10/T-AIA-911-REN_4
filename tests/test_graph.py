from pathlib import Path

import math

from src.graph.load_graph import Graph, load_graph
from src.graph.dijkstra import dijkstra


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_load_graph_contains_all_stations():
    stations_csv = DATA_DIR / "stations.csv"
    edges_csv = DATA_DIR / "edges.csv"

    graph = load_graph(str(stations_csv), str(edges_csv))

    # All station_ids from stations.csv should appear as keys in the graph.
    with stations_csv.open(encoding="utf-8") as f:
        header = f.readline()
        station_ids = [line.split(",")[0] for line in f if line.strip()]

    for station_id in station_ids:
        assert station_id in graph


def test_dijkstra_finds_direct_edge():
    # Minimal graph with a direct edge A -> B
    graph: Graph = {
        "A": [("B", 10.0)],
        "B": [],
    }

    path, distance = dijkstra(graph, "A", "B")

    assert path == ["A", "B"]
    assert distance == 10.0


def test_dijkstra_chooses_shortest_path():
    # Graph where A can reach C directly, but A->B->C is shorter
    graph: Graph = {
        "A": [("B", 3.0), ("C", 10.0)],
        "B": [("C", 4.0)],
        "C": [],
    }

    path, distance = dijkstra(graph, "A", "C")

    assert path == ["A", "B", "C"]
    assert distance == 7.0


def test_dijkstra_no_path_returns_inf():
    graph: Graph = {
        "A": [],
        "B": [],
    }

    path, distance = dijkstra(graph, "A", "B")

    assert path == []
    assert math.isinf(distance)

