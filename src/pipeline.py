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
from .graph.load_graph import Graph, load_graph
from .nlp.extract_stations import StationExtractionResult, extract_stations
# from .nlp.intent import Intent, detect_intent
# from .io.input_text import get_input_text



DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIONS_CSV = DATA_DIR / "stations.csv"
EDGES_CSV = DATA_DIR / "edges.csv"


def solve_travel_order(sentence: str) -> str:
    """Run the core pipeline on a given sentence and return a message.

    This helper is designed to be reused from other front-ends
    (CLI, Gradio app with speech-to-text, tests, etc.).
    """
    result = extract_stations(sentence)

    if result.error:
        return f"Extraction error: {result.error}"

    departure = result.departure
    arrival = result.arrival

    graph = load_graph(str(STATIONS_CSV), str(EDGES_CSV))
    path, distance = dijkstra(graph, departure, arrival)

    if not path:
        return f"No path found between {departure} and {arrival}."

    path_str = " -> ".join(path)
    return f"Shortest path: {path_str}\nTotal distance: {distance} km"


def run_pipeline() -> None:
    """Run the end-to-end processing pipeline for a single input.

    The function defines the high-level orchestration of the project
    without providing the concrete implementations of each step.
    """
    sentence = "Je veux aller de Rennes à Toulouse"
    print("Sentence:", sentence)

    message = solve_travel_order(sentence)
    print(message)



if __name__ == "__main__":
    run_pipeline()
