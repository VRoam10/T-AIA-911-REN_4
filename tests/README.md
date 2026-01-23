# Tests Overview

This folder contains unit tests for the travel order pipeline. All test docs and comments are in English by project rule.

## How to run

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

## Test modules

- `tests/test_graph.py`: validates CSV graph loading and Dijkstra shortest-path behavior.
- `tests/test_intent.py`: checks end-to-end intent classification for French travel vs non-travel vs non-French inputs.
- `tests/test_language_detection.py`: verifies French language detection for French, non-French, mixed, and short inputs.
- `tests/test_nlp_extract_stations.py`: tests station extraction for empty, known-city, and unknown-city sentences.
- `tests/test_travel_detection.py`: exercises travel-request detection patterns and intent mapping.
- `tests/test_coverage_complete.py`: covers remaining branches in graph, NLP, and pipeline modules to maintain 100% coverage.
