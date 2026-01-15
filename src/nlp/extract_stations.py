"""Station extraction from natural-language travel orders.

This module exposes a typed interface for extracting departure and
arrival stations from a user sentence. The goal is to offer a clean
contract where multiple NLP approaches can later be compared and
evaluated (e.g. simple heuristics, pattern-based methods, or more
advanced techniques).
"""

import csv
import re
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


@lru_cache(maxsize=1)
def _load_stations() -> Dict[str, str]:
    """Load station names and identifiers from the CSV file.

    The CSV is expected to live in the project-level ``data`` directory.
    The keys of the returned mapping are lowercased station names, and
    the values are their corresponding identifiers.
    """
    # ``.../src/nlp/extract_stations.py`` -> project root via parents[2]
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "stations.csv"

    stations: Dict[str, str] = {}
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("station_id") or "").strip()
                name = (row.get("station_name") or "").strip()
                if not code or not name:
                    continue
                stations[name.lower()] = code
    except OSError:
        # If the file cannot be read, fall back to an empty mapping.
        return {}

    return stations


def _find_station_codes(sentence: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (departure_code, arrival_code) detected in the sentence.

    The strategy is intentionally simple: it scans the sentence for
    occurrences of known station names (as whole words) and takes the
    first match as departure and the second as arrival.
    """
    text = sentence.lower()
    stations = _load_stations()

    matches: list[Tuple[int, str]] = []
    for name, code in stations.items():
        pattern = r"\b{}\b".format(re.escape(name))
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
