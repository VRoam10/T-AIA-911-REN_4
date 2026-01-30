"""Station extraction from natural-language travel orders.

This module exposes a typed interface for extracting departure and
arrival stations from a user sentence. The goal is to offer a clean
contract where multiple NLP approaches can later be compared and
evaluated (e.g. simple heuristics, pattern-based methods, or more
advanced techniques).
"""

import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Station:
    """A train station with its location."""

    code: str
    name: str
    lat: float
    lon: float


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


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance in km between two GPS coordinates.

    Uses the Haversine formula for accurate distance on Earth's surface.
    """
    R = 6371  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


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


@lru_cache(maxsize=1)
def _load_stations_with_coords() -> List[Station]:
    """Load stations with their GPS coordinates from the CSV file."""
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "stations.csv"

    stations: List[Station] = []
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("station_id") or "").strip()
                name = (row.get("station_name") or "").strip()
                lat_str = (row.get("lat") or "").strip()
                lon_str = (row.get("lon") or "").strip()
                if not code or not name or not lat_str or not lon_str:
                    continue
                stations.append(
                    Station(
                        code=code,
                        name=name,
                        lat=float(lat_str),
                        lon=float(lon_str),
                    )
                )
    except (OSError, ValueError):
        return []

    return stations


def find_nearest_station(lat: float, lon: float) -> Optional[Tuple[str, str, float]]:
    """Find the nearest station to the given GPS coordinates.

    Parameters
    ----------
    lat : float
        Latitude of the location
    lon : float
        Longitude of the location

    Returns
    -------
    Optional[Tuple[str, str, float]]
        A tuple of (station_code, station_name, distance_km) or None if no stations
    """
    stations = _load_stations_with_coords()
    if not stations:
        return None

    nearest = None
    min_distance = float("inf")

    for station in stations:
        distance = _haversine_distance(lat, lon, station.lat, station.lon)
        if distance < min_distance:
            min_distance = distance
            nearest = station

    if nearest:
        return (nearest.code, nearest.name, min_distance)
    return None


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
