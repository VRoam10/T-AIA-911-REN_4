# Architectural Patterns

This document describes recurring patterns observed across multiple files in the codebase.

## 1. Strategy Pattern

The codebase uses dictionaries to register interchangeable algorithms selected at runtime.

### NLP Strategy Registry
**Location**: `src/pipeline.py:36-45`

```python
NLP_STRATEGIES: Dict[str, StationExtractor] = {
    "rule_based": extract_stations,
    "legacy_spacy": spacy_extract_stations,
    "hf_ner": hf_extract_stations,
}
```

### Strategy Type Aliases
**Location**: `src/strategies.py:45-46`
- `CityStrategy = Literal["legacy_spacy", "hf_ner"]`
- `DateStrategy = Literal["eds", "hf_ner"]`

### Strategy Routing
**Location**: `src/strategies.py:178-212`
- `extract_locations_by_strategy()` - routes city extraction
- `extract_dates_by_strategy()` - routes date extraction

---

## 2. Singleton / Lazy Initialization Pattern

Global caches initialized on first use to avoid repeated expensive operations.

### Model Caching (ASR)
**Location**: `src/asr.py:23-72`
- Line 23: `MODEL_CACHE: Dict[Tuple[str, str, str], WhisperModel] = {}`
- Lines 49-66: `get_model()` with cache lookup and lazy load
- Lines 70-72: `clear_model_cache()` for cleanup

### NLP Pipeline Caching
**Location**: `src/nlp/hf_ner/ner.py:16-114`
- Line 16: `_PIPELINES: Dict[str, Any] = {}`
- Lines 90-114: `_get_pipe()` with cache check

### Geocoding Cache
**Location**: `src/strategies.py:64-98`
- Line 64: `_GEOCODE_CACHE: Dict[str, Any] = {}`
- Lines 82-98: `_cached_geocode()` pattern

### Station Data Caching (LRU)
**Location**: `src/nlp/extract_stations.py:80-146`
- Line 80: `@lru_cache(maxsize=1)` on `_load_stations()`
- Line 118: `@lru_cache(maxsize=1)` on `_load_stations_with_coords()`

---

## 3. Dataclass / TypedDict for Structured Results

Consistent use of typed containers for passing data between components.

### StationExtractionResult
**Location**: `src/nlp/extract_stations.py:30-48`
```python
@dataclass
class StationExtractionResult:
    departure_id: str | None
    arrival_id: str | None
    error: str | None
```

### ExtractionResult TypedDict
**Location**: `src/strategies.py:35-43`
```python
class ExtractionResult(TypedDict):
    locations_raw: List[str]
    cities: List[str]
    dates_raw: List[str]
    dates_norm: List[str]
```

### RouteDict TypedDict
**Location**: `src/strategies.py:27-32`

### StationPoint (frozen dataclass)
**Location**: `src/viz/map.py:13-18`

---

## 4. Enum-based State Classification

Closed set of states using Enum with auto() for type-safe intent handling.

**Location**: `src/nlp/intent.py:30-48`
```python
class Intent(Enum):
    UNKNOWN = auto()
    NOT_FRENCH = auto()
    NOT_TRIP = auto()
    TRIP = auto()
```

---

## 5. Defensive Dictionary Access

Consistent pattern using `.get()` with defaults and null coalescing for CSV parsing.

### CSV Row Parsing Examples
- `src/viz/map.py:31-40`: `row.get("field") or "".strip()`
- `src/nlp/extract_stations.py:96-98`: Same pattern
- `src/nlp/hf_ner/ner.py:126-128`: `e.get("entity_group") or e.get("entity")` fallback

---

## 6. Graceful Fallback with Exception Handling

Exceptions are caught and fallback strategies or defaults are returned.

### HF NER to SpaCy Fallback
**Location**: `src/strategies.py:188-211`
```python
try:
    # HF extraction
except Exception:
    return extract_locations_by_strategy(text, "legacy_spacy")
```

### Optional Dependency Imports
**Location**: `src/asr.py:12-20`
```python
try:
    import soundfile as sf
except ImportError:
    sf = None
```

### Device Fallback (GPU to CPU)
**Location**: `src/asr.py:57-63` - Try CUDA, fallback to CPU/int8

### Geocoding Exception Swallowing
**Location**: `src/strategies.py:54-61` - RateLimiter with `swallow_exceptions=True`

---

## 7. Pipeline Orchestration

Central function coordinates components in sequence, passing results between stages.

### Main Pipeline
**Location**: `src/pipeline.py:48-125`

`solve_travel_order()` orchestrates:
1. Intent detection (line 64)
2. NLP strategy selection (line 85)
3. Graph loading (line 103)
4. Path finding (line 104)
5. Map generation (lines 110-122)

### Extraction Pipeline
**Location**: `src/strategies.py:215-249`

`run_extraction()` chains:
1. Location extraction (line 234)
2. City validation (line 235)
3. Date extraction (line 237)
4. Date normalization (lines 240-242)

---

## 8. Private Function Convention

Internal/helper functions use underscore prefix.

### Examples
- `src/nlp/intent.py:51-149`: `_is_french()`, `_basic_french_detection()`
- `src/nlp/extract_stations.py:51-77`: `_canonicalize()`, `_haversine_distance()`
- `src/strategies.py:70-79`: `_dedupe_keep_order()`

---

## 9. CSV Data Loading Pattern

Consistent approach for loading and caching CSV resources with dataclass factories.

### Station CSV Loading
- `src/nlp/extract_stations.py:80-115`: Dict comprehension into cached mapping
- `src/nlp/extract_stations.py:118-146`: Dataclass object list
- `src/nlp/hf_ner/ner.py:21-47`: Module-level lazy init
- `src/viz/map.py:20-43`: Frozen dataclass factory

---

## 10. Module-Level Configuration Constants

Immutable configuration defined at module level.

### Path Constants
**Location**: `src/pipeline.py:26-29`
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STATIONS_CSV = DATA_DIR / "stations.csv"
EDGES_CSV = DATA_DIR / "edges.csv"
```

### Keyword Sets
**Location**: `src/strategies.py:66-67`
```python
DEPART_WORDS: set[str] = {"de", "depuis", ...}
DEST_WORDS: set[str] = {"à", "vers", ...}
```

### Model IDs
**Location**: `src/nlp/hf_ner/ner.py:13-14`

---

## Summary Table

| Pattern | Primary Files | Category |
|---------|---------------|----------|
| Strategy Pattern | pipeline.py, strategies.py | Behavioral |
| Singleton/Lazy Init | asr.py, hf_ner/ner.py, strategies.py | Creational |
| Dataclass/TypedDict | extract_stations.py, strategies.py, viz/map.py | Structural |
| Enum Classification | nlp/intent.py | Structural |
| Graceful Fallback | strategies.py, asr.py, monitoring.py | Error Handling |
| Pipeline Orchestration | pipeline.py, strategies.py | Communication |
| CSV Loading Pattern | extract_stations.py, hf_ner/ner.py, viz/map.py | Data Access |
