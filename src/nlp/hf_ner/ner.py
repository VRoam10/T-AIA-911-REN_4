# ner.py
from __future__ import annotations

import re
from typing import Dict, List, Any, Optional, Tuple

from pathlib import Path

from transformers import pipeline

from src.nlp.extract_stations import StationExtractionResult

DEFAULT_NER_MODEL: str = "Jean-Baptiste/camembert-ner"
DEFAULT_NER_DATES_MODEL: str = "Jean-Baptiste/camembert-ner-with-dates"

_PIPELINES: Dict[str, Any] = {}

_STATION_MAP: Dict[str, str] | None = None


def _load_station_map() -> Dict[str, str]:
    """Load station names -> station_id mapping from data/stations.csv."""
    global _STATION_MAP
    if _STATION_MAP is not None:
        return _STATION_MAP

    project_root = Path(__file__).resolve().parents[3]
    csv_path = project_root / "data" / "stations.csv"

    stations: Dict[str, str] = {}
    try:
        with csv_path.open(encoding="utf-8") as f:
            # local import to avoid csv at module import time
            import csv

            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("station_id") or "").strip()
                name = (row.get("station_name") or "").strip()
                if not code or not name:
                    continue
                stations[name.lower()] = code
    except OSError:
        stations = {}

    _STATION_MAP = stations
    return stations


def _find_station_codes_from_locations(
    sentence: str, locations: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Map extracted location strings to station codes.
    Uses order of appearance in the sentence when possible.
    """
    stations = _load_station_map()
    if not stations:
        return None, None

    lowered = sentence.lower()
    matches: List[Tuple[int, str]] = []

    for loc in locations:
        key = loc.lower().strip()
        code = stations.get(key)
        if not code:
            continue
        pos = lowered.find(key)
        if pos != -1:
            matches.append((pos, code))

    if not matches:
        return None, None

    matches.sort(key=lambda item: item[0])
    first = matches[0][1]
    second = matches[1][1] if len(matches) > 1 else None
    return first, second


def _clean(s: str) -> str:
    """Light cleanup: trims spaces and removes surrounding punctuation."""
    s = s.strip()
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _get_pipe(model_id: str) -> Any:
    """
    Lazy-load and cache HF pipeline.

    Common failure reasons:
    - missing sentencepiece
    - incompatible transformers/tokenizers versions on Windows
    """
    if model_id in _PIPELINES:
        return _PIPELINES[model_id]

    try:
        ner = pipeline("token-classification", model=model_id,
                       aggregation_strategy="simple")
    except Exception as e:
        raise RuntimeError(
            f"HF NER init failed for {model_id}. "
            "Try:\n"
            "  pip install sentencepiece\n"
            "  pip install -U transformers tokenizers protobuf\n"
            f"Original error: {e}"
        )

    _PIPELINES[model_id] = ner
    return ner


def extract_locations_hf(text: str, model_id: str = DEFAULT_NER_MODEL) -> List[str]:
    """
    Extract locations (LOC/GPE) from text using a HF token-classification model.
    """
    ner = _get_pipe(model_id)
    ents = ner(text)

    out: set[str] = set()
    for e in ents:
        label = e.get("entity_group") or e.get("entity")
        if label in {"LOC", "GPE"}:
            val = _clean(str(e.get("word", "")))
            if val:
                out.add(val)

    return sorted(out)


def extract_dates_hf(text: str, model_id: str = DEFAULT_NER_DATES_MODEL) -> List[str]:
    """
    Extract dates from text using a HF token-classification model that emits DATE labels.
    """
    ner = _get_pipe(model_id)
    ents = ner(text)

    out: set[str] = set()
    for e in ents:
        label = e.get("entity_group") or e.get("entity")
        if label == "DATE":
            val = _clean(str(e.get("word", "")))
            if val:
                out.add(val)

    return sorted(out)


def extract_stations_hf(sentence: str) -> StationExtractionResult:
    """
    Extract departure/arrival station codes using HF NER locations.
    Falls back to error if not enough stations are found.
    """
    if not sentence or not sentence.strip():
        return StationExtractionResult(
            departure=None,
            arrival=None,
            error="Empty sentence.",
        )

    locations = extract_locations_hf(sentence)
    departure, arrival = _find_station_codes_from_locations(
        sentence, locations)

    error: Optional[str] = None
    if departure is None or arrival is None:
        error = "Could not detect both departure and arrival stations. from hf ner"

    return StationExtractionResult(
        departure=departure,
        arrival=arrival,
        error=error,
    )
