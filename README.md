# Travel Order Resolver

The **Travel Order Resolver** is an academic Python project whose goal is to
transform a free-form travel order (in natural language) into structured
information that can be used to compute and justify an itinerary.

At this stage, the focus is on designing a clear, modular and typed
architecture. All business logic (NLP, graph construction and path-finding) is
intentionally left unimplemented.

## Global Pipeline

The project is organized as a pipeline with four main stages:

1. **Input** – acquire a sentence describing a travel order (text now, possibly
   speech-to-text later).
2. **NLP** – detect the user's intent and extract departure/arrival stations.
3. **Graph** – load a transportation graph from CSV files into an in-memory
   structure.
4. **Output** – compute and prepare a candidate route (e.g. using Dijkstra).

The pipeline is orchestrated by `src/pipeline.py`, which wires these steps
together but delegates all real work to dedicated modules.

## Modular Architecture

The code is split into independent bricks, each with a well-defined interface:

- `src/io/` – input acquisition (`get_input_text`).
- `src/nlp/` – natural language processing:
  - `intent.py` defines the `Intent` enum and the `detect_intent` interface.
  - `extract_stations.py` defines the `StationExtractionResult` dataclass and
    the `extract_stations` interface.
- `src/graph/` – graph management:
  - `load_graph.py` defines the `Graph` type and the `load_graph` interface
    (CSV → graph).
  - `dijkstra.py` declares the `dijkstra` function used for shortest-path
    computation.
- `src/pipeline.py` – high-level orchestration of the pipeline.

Each module is fully typed and documented with docstrings, but all functions
raise `NotImplementedError`. This ensures that:

- the project imports cleanly,
- the architecture is stable and testable,
- the actual algorithms and heuristics can be introduced incrementally.

## Data and Tests

The `data/` directory contains placeholder CSV files:

- `data/stations.csv` – list of stations,
- `data/edges.csv` – edges between stations with an associated weight.

The `tests/` package is present but empty, ready to receive unit tests as soon
as the individual bricks (NLP, graph loading, Dijkstra, etc.) are implemented.

> Dataset fictif utilisé uniquement pour la baseline du projet.

## Next Steps

- Implement intent detection and station extraction strategies in `src/nlp/`.
- Implement CSV parsing and graph construction in `src/graph/load_graph.py`.
- Implement Dijkstra’s algorithm in `src/graph/dijkstra.py` and analyze its
  complexity.
- Add unit tests under `tests/` to validate each brick independently and the
  pipeline as a whole.
