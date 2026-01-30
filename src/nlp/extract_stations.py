"""Station extraction from natural-language travel orders.

This module exposes a typed interface for extracting departure and
arrival stations from a user sentence. The goal is to offer a clean
contract where multiple NLP approaches can later be compared and
evaluated (e.g. simple heuristics, pattern-based methods, or more
advanced techniques).
"""

import csv
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class StationExtractionResult:
    """Container for the result of station extraction.

    Attributes
    ----------
    departure:
        The name or identifier of the departure station, if detected.
    arrival:
        The name or identifier of the arrival station, if detected.
    error:
        An optional error message describing why extraction failed or
        is incomplete. This allows the pipeline to distinguish between
        missing information and processing errors.
    """

    departure: Optional[str]
    arrival: Optional[str]
    error: Optional[str]


def _canonicalize(text: str) -> str:
    """Normalize text for matching (remove accents/punctuation)."""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalized.strip()


@lru_cache(maxsize=1)
def _load_stations() -> Dict[str, str]:
    """Load station names and identifiers from the CSV file.

    The CSV lives in ``data/`` at the project root. The mapping keys are
    canonicalized strings (station name or city name without accents and
    punctuation) so the NLP layer can match more natural phrases.
    """
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "stations.csv"

    stations: Dict[str, str] = {}
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("station_id") or "").strip()
                name = (row.get("station_name") or "").strip()
                city = (row.get("city") or "").strip()

                if not code:
                    continue

                if name:
                    key = _canonicalize(name)
                    if key:
                        stations[key] = code

                if city:
                    key = _canonicalize(city)
                    if key:
                        stations[key] = code
    except OSError:
        return {}

    return stations


def _find_station_codes(sentence: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (departure_code, arrival_code) detected in the sentence.

    The strategy scans a canonicalized version of the sentence and looks
    for whole-word matches against the precomputed station keys.
    """
    text = _canonicalize(sentence)
    stations = _load_stations()

    matches: list[Tuple[int, str]] = []
    for normalized_name, code in stations.items():
        pattern = r"\b{}\b".format(re.escape(normalized_name))
        match = re.search(pattern, text)
        if match:
            matches.append((match.start(), code))

    if not matches:
        return None, None

    matches.sort(key=lambda item: item[0])
    first = matches[0][1]
    second = matches[1][1] if len(matches) > 1 else None
    return first, second


def extract_stations(sentence: str) -> StationExtractionResult:
    """Extract departure and arrival stations from a sentence.

    Parameters
    ----------
    sentence:
        The raw input sentence describing a potential travel order.

    Returns
    -------
    StationExtractionResult
        A structured result containing the extracted stations and any
        associated error information.

    Notes
    -----
    This function is the main entry point for experimenting with
    multiple NLP strategies. Different implementations can be plugged
    in and compared (for example rule-based extraction versus
    tokenization and tagging), while keeping the rest of the pipeline
    unchanged.
    """
    if not sentence or not sentence.strip():
        return StationExtractionResult(
            departure=None,
            arrival=None,
            error="Empty sentence.",
        )

    departure, arrival = _find_station_codes(sentence)

    error: Optional[str] = None
    if departure is None or arrival is None:
        error = "Could not detect both departure and arrival stations."

    return StationExtractionResult(
        departure=departure,
        arrival=arrival,
        error=error,
    )
