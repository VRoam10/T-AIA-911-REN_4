"""Graph loading from CSV files.

This module defines the Graph type used throughout the project and the
interface for loading it from tabular data stored in CSV files.
"""

import csv
from typing import Dict, List, Tuple

Graph = Dict[str, List[Tuple[str, float]]]

def load_graph(stations_path: str, edges_path: str) -> Graph:
    graph: Graph = {}

    # 1) Initialiser tous les sommets à partir de stations.csv
    with open(stations_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = row["station_id"]
            graph[station_id] = []

    # 2) Ajouter les arêtes à partir de edges.csv
    with open(edges_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_id = row["from_station_id"]
            to_id = row["to_station_id"]
            distance = float(row["distance_km"])
            # on ajoute l’arête orientée from_id -> to_id
            graph.setdefault(from_id, []).append((to_id, distance))

    return graph
