from __future__ import annotations

from typing import List

import spacy

from src.nlp.extract_stations import (
    StationExtractionResult,
    extract_stations_from_locations,
)

_nlp = None


def _get_nlp():
    """Lazy-load the SpaCy model with eds.dates pipeline."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("fr_core_news_md")
        _nlp.add_pipe("eds.dates")
    return _nlp


def extract_locations_spacy(text: str) -> List[str]:
    """
    Extract location entities (LOC, GPE) using spaCy FR model.

    Pros: fast (local), no extra deps
    Cons: less robust on ASR transcripts (punctuation/casing noise)
    """
    doc = _get_nlp()(text)
    return sorted({ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE")})


def extract_dates_eds(text: str) -> List[str]:
    """
    Extract date expressions using eds.dates pipeline (rule-based).

    Pros: fast, no HF dependency
    Cons: may miss natural/colloquial forms, can be noisy
    """
    doc = _get_nlp()(text)
    spans = doc.spans.get("dates", [])
    return [sp.text for sp in spans]


def extract_stations_spacy(sentence: str) -> StationExtractionResult:
    """Extract stations from a sentence using spaCy NER locations."""
    locations = extract_locations_spacy(sentence)
    return extract_stations_from_locations(sentence, locations)
