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
from typing import Callable, Dict, Optional, Tuple, Union

from .graph.dijkstra import dijkstra
from .graph.load_graph import Graph, load_graph
from .nlp.extract_stations import StationExtractionResult, extract_stations
from .nlp.hf_ner import extract_stations_hf
from .nlp.intent import Intent, detect_intent
from .nlp.legacy_spacy import extract_stations_spacy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATIONS_CSV = DATA_DIR / "stations.csv"
EDGES_CSV = DATA_DIR / "edges.csv"


# Simple strategy registries so we can swap NLP / path-finding
StationExtractor = Callable[[str], StationExtractionResult]
PathFinder = Callable[[Graph, str, str], Tuple[list[str], float]]

NLP_STRATEGIES: Dict[str, StationExtractor] = {
    "rule_based": extract_stations,
    "legacy_spacy": extract_stations_spacy,
    "hf_ner": extract_stations_hf,
}

PATH_FINDER_STRATEGIES: Dict[str, PathFinder] = {
    "dijkstra": dijkstra,
    # "dijkstra": dijkstra2,
}


def solve_travel_order(
    sentence: str,
    nlp_name: str = "rule_based",
    path_name: str = "dijkstra",
    *,
    departure_station: Optional[str] = None,
    arrival_station: Optional[str] = None,
    generate_map: bool = True,
    map_output_html: Optional[Union[str, Path]] = None,
) -> str:
    """Run the core pipeline on a given sentence and return a message.

    This helper is designed to be reused from other front-ends
    (CLI, Gradio app with speech-to-text, tests, etc.).
    """
    # Step 1: Detect intent first
    intent = detect_intent(sentence)

    if intent == Intent.UNKNOWN:
        return "Error: Empty or invalid input"
    elif intent == Intent.NOT_FRENCH:
        return "Error: Input is not in French"
    elif intent == Intent.NOT_TRIP:
        return "Error: Input is not a travel request"

    # Step 2: If intent is TRIP, proceed with NLP extraction
    path_finder = PATH_FINDER_STRATEGIES.get(path_name)
    if path_finder is None:
        return f"Unknown path-finding strategy: {path_name!r}"

    departure: Optional[str] = None
    arrival: Optional[str] = None

    if departure_station and arrival_station:
        departure = departure_station
        arrival = arrival_station
    else:
        nlp = NLP_STRATEGIES.get(nlp_name)
        if nlp is None:
            return f"Unknown NLP strategy: {nlp_name!r}"

        result = nlp(sentence)

        if result.error:
            return f"Extraction error: {result.error}"

        if result.departure is None or result.arrival is None:
            raise ValueError("Departure or arrival not set")

        departure = result.departure
        arrival = result.arrival

    if departure is None or arrival is None:
        raise ValueError("Departure or arrival not set")

    graph = load_graph(str(STATIONS_CSV), str(EDGES_CSV))
    path, distance = path_finder(graph, departure, arrival)

    if not path:
        return f"No path found between {departure} and {arrival}."

    map_message = ""
    if generate_map:
        try:
            from .viz.map import plot_path

            output_path = (
                Path(map_output_html)
                if map_output_html is not None
                else PROJECT_ROOT / "trajectory.html"
            )
            plot_path(path, STATIONS_CSV, output_path)
            map_message = f"\nMap saved to: {output_path}"
        except Exception as exc:
            map_message = f"\nMap generation failed: {exc}"

    path_str = " -> ".join(path)
    return f"Shortest path: {path_str}\nTotal distance: {distance} km{map_message}"


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
