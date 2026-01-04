"""Graph loading from CSV files.

This module defines the Graph type used throughout the project and the
interface for loading it from tabular data stored in CSV files.
"""

from typing import Dict, List, Tuple

Graph = Dict[str, List[Tuple[str, float]]]
"""Type alias for the transportation graph.

Each key represents a station identifier, and the associated value is a
list of `(neighbor_station, weight)` pairs, where the weight typically
corresponds to a distance, duration, or cost.
"""


def load_graph(stations_path: str, edges_path: str) -> Graph:
    """Load a transportation graph from CSV files.

    Parameters
    ----------
    stations_path:
        Path to the CSV file describing the set of stations.
    edges_path:
        Path to the CSV file describing connections between stations,
        along with their associated weights.

    Returns
    -------
    Graph
        An in-memory adjacency-list representation of the network.

    Notes
    -----
    The implementation will parse the CSV files and transform them into
    the `Graph` structure defined above. At this stage, only the
    interface and typing are provided; the actual CSV-to-graph
    transformation will be implemented later.
    """
    raise NotImplementedError("Graph loading is not implemented yet.")

