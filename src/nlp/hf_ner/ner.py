# ner.py
from __future__ import annotations

import re
from typing import Any, Dict, List

from transformers import pipeline

from src.nlp.extract_stations import (
    StationExtractionResult,
    extract_stations_from_locations,
)

DEFAULT_NER_MODEL: str = "Jean-Baptiste/camembert-ner"
DEFAULT_NER_DATES_MODEL: str = "Jean-Baptiste/camembert-ner-with-dates"

_PIPELINES: Dict[str, Any] = {}


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
        ner = pipeline(
            "token-classification", model=model_id, aggregation_strategy="simple"
        )
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
    result = extract_stations_from_locations(sentence, locations)
    if result.error:
        return StationExtractionResult(
            departure=result.departure,
            arrival=result.arrival,
            error=f"{result.error} (hf ner)",
        )
    return result
