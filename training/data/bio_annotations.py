"""BIO annotation helpers for NER training data.

Converts sentences with known departure/arrival stations into
BIO-tagged token sequences for token classification training.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Optional


def _canonicalize(text: str) -> str:
    """Normalize text for matching (remove accents/punctuation, lowercase)."""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalized.strip()


def load_station_names(stations_csv: Path) -> dict[str, list[str]]:
    """Load station names and city names from CSV.

    Returns:
        Dict mapping canonicalized name -> list of original names.
    """
    names: dict[str, list[str]] = {}
    with open(stations_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_name = row["station_name"].strip()
            city = row.get("city", "").strip()

            canon_station = _canonicalize(station_name)
            canon_city = _canonicalize(city)

            if canon_station:
                names.setdefault(canon_station, []).append(station_name)
            if canon_city and canon_city != canon_station:
                names.setdefault(canon_city, []).append(city)

    return names


def tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer preserving token boundaries.

    Splits on whitespace and separates punctuation like apostrophes.
    """
    tokens = []
    for word in text.split():
        # Split on apostrophes: "d'antibes" -> ["d'", "antibes"]
        parts = re.split(r"(?<=')(?=[a-zA-ZÀ-ÿ])", word)
        for part in parts:
            # Strip trailing punctuation but keep it if it's all punctuation
            clean = part.rstrip(".,;:!?")
            if clean:
                tokens.append(clean)
            elif part:
                tokens.append(part)
    return tokens


def find_span_in_tokens(tokens: list[str], target: str) -> Optional[tuple[int, int]]:
    """Find the span (start, end) of a target string in token list.

    Tries to match the target against consecutive tokens.
    Returns (start_idx, end_idx_exclusive) or None.
    """
    target_canon = _canonicalize(target)
    if not target_canon:
        return None

    target_words = target_canon.split()
    n = len(target_words)

    for i in range(len(tokens) - n + 1):
        window = " ".join(_canonicalize(t) for t in tokens[i : i + n])
        if window == target_canon:
            return (i, i + n)

    # Fallback: try partial matching for single-word targets
    if len(target_words) == 1:
        for i, token in enumerate(tokens):
            if _canonicalize(token) == target_canon:
                return (i, i + 1)

    return None


def annotate_bio(
    sentence: str,
    departure_name: Optional[str],
    arrival_name: Optional[str],
) -> tuple[list[str], list[str]]:
    """Create BIO annotations for a sentence.

    Args:
        sentence: The input sentence.
        departure_name: The departure station/city name found in the sentence.
        arrival_name: The arrival station/city name found in the sentence.

    Returns:
        Tuple of (tokens, bio_tags) where bio_tags is a list of
        "O", "B-DEP", "I-DEP", "B-ARR", "I-ARR" strings.
    """
    tokens = tokenize_simple(sentence)
    tags = ["O"] * len(tokens)

    if not tokens:
        return tokens, tags

    # Find departure span
    dep_span = None
    if departure_name:
        dep_span = find_span_in_tokens(tokens, departure_name)
        if dep_span:
            start, end = dep_span
            tags[start] = "B-DEP"
            for j in range(start + 1, end):
                tags[j] = "I-DEP"

    # Find arrival span (avoid overlapping with departure)
    if arrival_name:
        arr_span = find_span_in_tokens(tokens, arrival_name)
        if arr_span:
            start, end = arr_span
            # Check no overlap with departure
            if dep_span is None or start >= dep_span[1] or end <= dep_span[0]:
                tags[start] = "B-ARR"
                for j in range(start + 1, end):
                    tags[j] = "I-ARR"

    return tokens, tags
