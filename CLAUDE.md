# Travel Order Resolver

French travel request processor that extracts departure/arrival stations from natural language and computes optimal routes using Dijkstra's algorithm.

## Tech Stack

- **Language**: Python 3.11+
- **Architecture**: Hexagonal (Ports & Adapters) with DI container
- **NLP**: SpaCy 3.8, HuggingFace Transformers, EDS-NLP
- **Speech-to-Text**: Faster-Whisper (OpenAI Whisper variant)
- **Web UI**: Gradio 6.2
- **Visualization**: Folium (maps), ReportLab (PDF)
- **Config**: Pydantic Settings (env vars support)
- **Dev Tools**: Black, isort, mypy, pytest, pre-commit

## Project Structure

```
src/
├── config.py                # Centralized Pydantic Settings
├── container.py             # DI container
├── domain/
│   ├── models.py           # Frozen dataclasses (immutable)
│   └── errors.py           # Typed domain errors
├── ports/                   # Abstract interfaces (Protocols)
│   ├── nlp.py, graph.py, geocoding.py, asr.py, rendering.py, cache.py
├── adapters/                # Concrete implementations
│   ├── nlp/                # RuleBasedExtractor, SpaCyNERAdapter, HFNERAdapter
│   ├── graph/              # CSVGraphRepository, DijkstraRouteSolver
│   ├── geocoding/          # NominatimGeocoderAdapter
│   ├── asr/                # WhisperASRAdapter
│   ├── rendering/          # FoliumMapRenderer
│   └── cache/              # InMemoryCache, NullCache
├── services/                # Application orchestration
│   ├── travel_resolver.py  # Main service
│   └── extraction_service.py
├── pipeline.py              # Legacy orchestrator (still works)
├── nlp/, graph/, viz/       # Legacy modules (wrapped by adapters)
```

## Essential Commands

```bash
# Installation
pip install -r requirements-dev.txt
python -m spacy download fr_core_news_md
pre-commit install --hook-type commit-msg --hook-type pre-commit

# Run (new architecture)
python -c "from src.container import Container; ..."  # See examples below

# Run (legacy - still works)
python -m src.pipeline          # Demo pipeline
python -m apps.app              # Gradio UI

# Test
pytest                                          # All tests (73 pass)
pytest --cov=src --cov-report=term-missing     # With coverage

# Lint
black . && isort . && mypy .
```

## New Architecture Usage

```python
from src.container import Container
from src.services import TravelResolverService

# Create container with all dependencies
container = Container.create_default()

# Resolve service
service = container.resolve(TravelResolverService)

# Use service
result, error = service.resolve_safe("Je veux aller de Rennes à Toulouse")
if not error:
    print(f"Route: {result.path}")
    print(f"Distance: {result.total_distance_km} km")
```

## Key Entry Points (New)

| Purpose | Location |
|---------|----------|
| DI Container | `src/container.py` - `Container.create_default()` |
| Main Service | `src/services/travel_resolver.py` - `TravelResolverService` |
| Config | `src/config.py` - `get_config()` |
| Domain Models | `src/domain/models.py` - `StationExtractionResult`, `RouteResult` |
| Ports | `src/ports/*.py` - Protocol interfaces |

## Key Entry Points (Legacy)

| Purpose | Location |
|---------|----------|
| Pipeline orchestration | `src/pipeline.py:48` - `solve_travel_order()` |
| Intent classification | `src/nlp/intent.py:588` - `detect_intent()` |
| Station extraction | `src/nlp/extract_stations.py:229` - `extract_stations()` |

## Configuration (Environment Variables)

```bash
TOR_NLP_DEFAULT_STRATEGY=rule_based  # or hf_ner, spacy
TOR_ASR_DEVICE=cuda                   # or cpu, auto
TOR_GRAPH_DATA_DIR=/path/to/data
TOR_GEO_TIMEOUT_SECONDS=10
TOR_LOG_LEVEL=INFO
```

## Critical Fixes in New Architecture

1. **SpaCy lazy loading** - Model no longer loaded at import time (`adapters/nlp/spacy_adapter.py`)
2. **Cache invalidation** - All caches can be cleared for testing (`InMemoryCache.clear()`)
3. **Logged fallbacks** - No more silent exception swallowing (`services/extraction_service.py`)
4. **ASR cache key fix** - Cache key uses actual device after fallback (`adapters/asr/whisper_adapter.py`)

## Additional Documentation

| Topic | File |
|-------|------|
| **Technical documentation** | `docs/TECHNICAL.md` |
| **Usage guide** | `docs/USAGE.md` |
| Architectural patterns | `.claude/docs/architectural_patterns.md` |
| Testing details | `tests/README.md` |
| Developer guidelines | `AGENTS.md` |

## CI/CD

GitHub Actions runs on PRs: Black, isort, mypy, pytest
