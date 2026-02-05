# strategies.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, Optional, Tuple, TypedDict

import requests
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from src.dates import normalize_dates_fr
from src.nlp.hf_ner import extract_dates_hf, extract_locations_hf
from src.nlp.legacy_spacy import extract_dates_eds, extract_locations_spacy

# ===================== Types =====================


class CityDict(TypedDict):
    name: str
    lat: float
    lon: float
    address: Dict[str, Any]
    station_code: NotRequired[str]
    station_name: NotRequired[str]
    station_distance_km: NotRequired[float]


RouteDict = TypedDict(
    "RouteDict",
    {
        "depart": Optional[CityDict],
        "destinations": List[CityDict],
    },
)

ExtractionResult = TypedDict(
    "ExtractionResult",
    {
        "locations_raw": List[str],
        "cities": List[CityDict],
        "dates_raw": List[str],
        "dates_norm": Optional[List[str]],
    },
)

CityStrategy = Literal["legacy_spacy", "hf_ner"]
DateStrategy = Literal["eds", "hf_ner"]

# ===================== Geocoder init (robust) =====================
# Important: increase timeout. Default small timeouts cause frequent failures.
_session = requests.Session()
geolocator = Nominatim(user_agent="whisper-app", timeout=10)
geolocator.session = _session

# RateLimiter: swallows exceptions so your app doesn't crash when Nominatim is down.
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.0,  # be nice to the public endpoint
    max_retries=2,
    error_wait_seconds=2.0,
    swallow_exceptions=True,
)

# Simple in-memory cache so "Paris" is not requested 50 times per run
_GEOCODE_CACHE: Dict[str, Any] = {}

DEPART_WORDS = {"de", "depuis", "quitter"}
DEST_WORDS = {"à", "vers", "pour", "en direction de"}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """Remove duplicates while keeping the original order."""
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        x2 = x.strip()
        if x2 and x2 not in seen:
            out.append(x2)
            seen.add(x2)
    return out


def _cached_geocode(query: str) -> Any:
    """
    Cached geocode call (Nominatim).
    Returns geopy Location or None.
    """
    key = query.strip().lower()
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]

    # swallow_exceptions=True already prevents crashes, but we keep a defensive try/except
    try:
        res = geocode(query, language="fr", addressdetails=True)
    except (GeocoderTimedOut, GeocoderUnavailable, Exception):
        res = None

    _GEOCODE_CACHE[key] = res
    return res


def extract_valid_cities(raw_places: List[str]) -> List[CityDict]:
    """
    Validate raw place strings by geocoding and keeping only places that map to a city/town/village.

    This reduces false positives from NER.
    """
    cities: List[CityDict] = []
    for place in raw_places:
        location = _cached_geocode(place)
        if not location:
            continue

        address = location.raw.get("address", {})
        city_name = (
            address.get("city")
            or address.get("municipality")
            or address.get("town")
            or address.get("village")
        )
        if not city_name:
            continue

        cities.append(
            {
                "name": str(city_name),
                "lat": float(location.latitude),
                "lon": float(location.longitude),
                "address": dict(address),
            }
        )
    return cities


def extract_departure_and_destinations(text: str, cities: List[CityDict]) -> RouteDict:
    """
    Heuristic route extraction:
    - Detects a "depart" city using keywords: {"de","depuis","quitter"}
    - Detects "destinations" using keywords: {"à","vers","pour","en direction de"}
    - If no depart keyword found, picks the first city mentioned in the text.
    """
    lowered = text.lower()
    route: RouteDict = {"depart": None, "destinations": []}

    # Order cities by appearance in text (best-effort)
    city_positions: List[Tuple[int, CityDict]] = []
    for city in cities:
        pos = lowered.find(city["name"].lower())
        if pos != -1:
            city_positions.append((pos, city))
    city_positions.sort(key=lambda x: x[0])
    ordered_cities = [c for _, c in city_positions]

    if not ordered_cities:
        return route

    # Find departure
    for city in ordered_cities:
        name_lower = city["name"].lower()
        if any(f"{kw} {name_lower}" in lowered for kw in DEPART_WORDS):
            route["depart"] = city
            break

    # Find destinations
    for city in ordered_cities:
        if route["depart"] and city["name"] == route["depart"]["name"]:
            continue
        name_lower = city["name"].lower()
        if any(f"{kw} {name_lower}" in lowered for kw in DEST_WORDS):
            route["destinations"].append(city)

    # Fallback: first city as departure if none found
    if not route["depart"]:
        route["depart"] = ordered_cities[0]

    return route


def extract_locations_by_strategy(text: str, strategy: CityStrategy) -> List[str]:
    """
    Strategy router for location extraction.

    - legacy_spacy: spaCy only
    - hf_ner: HF CamemBERT NER only (requires transformers + sentencepiece)
    """
    if strategy == "legacy_spacy":
        return extract_locations_spacy(text)

    if strategy == "hf_ner":
        try:
            return extract_locations_hf(text)
        except Exception:
            # If HF fails (tokenizer/version), fallback to spaCy
            return extract_locations_spacy(text)
    raise ValueError(f"Unknown city strategy: {strategy!r}")


def extract_dates_by_strategy(text: str, strategy: DateStrategy) -> List[str]:
    """
    Strategy router for date extraction.

    - eds: eds.dates only (rule-based)
    - hf_ner: HF NER with DATE label (may fail if versions mismatch)
    """
    if strategy == "eds":
        return extract_dates_eds(text)

    if strategy == "hf_ner":
        try:
            return extract_dates_hf(text)
        except Exception:
            return extract_dates_eds(text)
    raise ValueError(f"Unknown date strategy: {strategy!r}")


def run_extraction(
    full_text: str,
    city_strategy: CityStrategy,
    date_strategy: DateStrategy,
    dates_normalize: bool = True,
) -> ExtractionResult:
    """
    End-to-end extraction:
    1) Extract raw locations using selected strategy
    2) Geocode + validate into cities
    3) Extract dates using selected strategy
    4) Optional: normalize dates into ISO (YYYY-MM-DD)

    Returns:
      - locations_raw: raw place strings
      - cities: validated cities with lat/lon/address
      - dates_raw: raw date strings (as found)
      - dates_norm: ISO dates (if enabled)
    """
    locs = extract_locations_by_strategy(full_text, city_strategy)
    cities = extract_valid_cities(locs)

    dates_raw = extract_dates_by_strategy(full_text, date_strategy)
    dates_raw = _dedupe_keep_order(dates_raw)

    dates_norm: Optional[List[str]] = None
    if dates_normalize:
        dates_norm = normalize_dates_fr(dates_raw) if dates_raw else []

    return {
        "locations_raw": locs,
        "cities": cities,
        "dates_raw": dates_raw,
        "dates_norm": dates_norm,
    }
