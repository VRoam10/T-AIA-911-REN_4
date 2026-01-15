"""High-level pipeline orchestration for the Travel Order Resolver.

The pipeline is organized in several stages:

1. Input acquisition (text or, later, speech-to-text).
2. NLP processing (intent detection and station extraction).
3. Graph loading (from CSV files to an in-memory structure).
4. Path computation (e.g. via Dijkstra's algorithm).

This module wires these stages together without implementing any
business logic. Each step delegates work to dedicated, testable
modules.
"""

from pathlib import Path

from .graph.dijkstra import dijkstra
from .graph.load_graph import load_graph
from .nlp.extract_stations import extract_stations

# from .nlp.intent import Intent, detect_intent
# from .io.input_text import get_input_text


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIONS_CSV = DATA_DIR / "stations.csv"
EDGES_CSV = DATA_DIR / "edges.csv"


def run_pipeline() -> None:
    """Run the end-to-end processing pipeline for a single input.

    The function defines the high-level orchestration of the project
    without providing the concrete implementations of each step.
    """

    sentence = "Je veux aller de Rennes à Toulouse"
    print("Sentence:", sentence)

    result = extract_stations(sentence)

    if result.error or not result.departure or not result.arrival:
        print("Extraction error:", result.error)
        return

    departure = result.departure
    arrival = result.arrival

    print("Departure:", departure)
    print("Arrival:", arrival)

    graph = load_graph(str(STATIONS_CSV), str(EDGES_CSV))

    path, distance = dijkstra(graph, departure, arrival)

    if not path:
        print("No path found.")
        return

    print("Shortest path:", " -> ".join(path))
    print("Total distance:", distance, "km")


if __name__ == "__main__":
    run_pipeline()
