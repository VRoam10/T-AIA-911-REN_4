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
from .io.input_text import get_input_text
from .nlp.extract_stations import StationExtractionResult, extract_stations
from .nlp.intent import Intent, detect_intent


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIONS_CSV = DATA_DIR / "stations.csv"
EDGES_CSV = DATA_DIR / "edges.csv"


def run_pipeline() -> None:
    """Run the end-to-end processing pipeline for a single input.

    The function defines the high-level orchestration of the project
    without providing the concrete implementations of each step.
    """
    # Acquire raw input text.
    text = get_input_text()

    # Detect the intent behind the sentence.
    intent: Intent = detect_intent(text)

    # Extract stations from the sentence.
    stations: StationExtractionResult = extract_stations(text)

    # Load the transportation graph from CSV files.
    graph: Graph = load_graph(str(STATIONS_CSV), str(EDGES_CSV))

    # Compute a route between the extracted stations.
    # Error handling and validation of extracted data will be added
    # together with the future implementations.
    _path, _cost = dijkstra(
        graph=graph,
        start=stations.departure or "",
        end=stations.arrival or "",
    )

    # The final stage (formatting and presenting the result) is
    # intentionally left undefined at this point.


if __name__ == "__main__":
    run_pipeline()

