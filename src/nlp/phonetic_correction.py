"""Phonetic correction for speech-to-text errors with context awareness.

This module corrects common transcription errors from Whisper STT
when users say city names that are transcribed incorrectly due to
phonetic similarity. Uses rapidfuzz for intelligent fuzzy matching
instead of manual dictionaries.

Example
-------
    >>> correct_city_names("Je veux aller de Reine à Lion")
    "Je veux aller de Rennes à Lyon"
    >>> correct_city_names("La reine de France")
    "La reine de France"  # No correction (wrong context)
"""

import csv
import re
from pathlib import Path
from typing import List, Optional

from rapidfuzz import fuzz, process

# Travel context keywords that indicate we're talking about a journey
TRAVEL_KEYWORDS = [
    "aller",
    "veux aller",
    "vais",
    "trajet",
    "itinéraire",
    "voyage",
    "partir",
    "direction",
    "rejoindre",
    "rendre",
    "se rendre",
    "comment aller",
    "pour aller",
    "chemin",
    "route",
    "depuis",
    "vers",
    "pour",
]

# Prepositions that indicate location/destination
LOCATION_PREPOSITIONS = [
    "à",
    "de",
    "depuis",
    "vers",
    "pour",
    "en direction de",
]

# Minimum similarity score (0-100) to consider a match
MIN_SIMILARITY_SCORE = 70

# Manual corrections for difficult cases that fuzzy matching misses
MANUAL_CORRECTIONS = {
    "rince": "Reims",
    "rains": "Reims",
    "reim": "Reims",
    "bordo": "Bordeaux",
}

# Cache for city names loaded from CSV
_CITY_NAMES: Optional[List[str]] = None

# Cache for all French cities (not just stations)
_ALL_FRENCH_CITIES: Optional[set] = None


def _load_all_french_cities() -> set:
    """Load all French city names from french_cities.txt.

    This list is used to prevent false corrections of valid French cities
    that are not in our stations database.

    Returns
    -------
    set
        Set of all French city names (lowercase for comparison)
    """
    global _ALL_FRENCH_CITIES

    if _ALL_FRENCH_CITIES is not None:
        return _ALL_FRENCH_CITIES

    # Get path to french_cities.txt
    current_file = Path(__file__).resolve()
    data_dir = current_file.parent.parent.parent / "data"
    cities_txt = data_dir / "french_cities.txt"

    cities = set()
    try:
        with open(cities_txt, "r", encoding="utf-8") as f:
            for line in f:
                city = line.strip()
                if city:  # Skip empty lines
                    cities.add(city.lower())
    except Exception as e:
        print(f"Warning: Could not load french_cities.txt: {e}")

    _ALL_FRENCH_CITIES = cities
    return cities


def _load_city_names() -> List[str]:
    """Load city names from stations.csv.

    Returns
    -------
    List[str]
        List of city names
    """
    global _CITY_NAMES

    if _CITY_NAMES is not None:
        return _CITY_NAMES

    # Get path to stations.csv
    current_file = Path(__file__).resolve()
    data_dir = current_file.parent.parent.parent / "data"
    stations_csv = data_dir / "stations.csv"

    cities = []
    try:
        with open(stations_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "station_name" in row and row["station_name"]:
                    cities.append(row["station_name"])
    except Exception as e:
        print(f"Warning: Could not load stations.csv: {e}")
        # Fallback to common cities if CSV loading fails
        cities = [
            "Paris",
            "Lyon",
            "Marseille",
            "Toulouse",
            "Nice",
            "Nantes",
            "Strasbourg",
            "Montpellier",
            "Bordeaux",
            "Lille",
            "Rennes",
            "Reims",
            "Le Havre",
            "Saint-Étienne",
            "Toulon",
            "Grenoble",
            "Dijon",
            "Angers",
            "Nîmes",
            "Villeurbanne",
        ]

    _CITY_NAMES = cities
    return cities


def _has_travel_context(text: str) -> bool:
    """Check if the text contains travel-related keywords.

    Parameters
    ----------
    text : str
        The text to analyze

    Returns
    -------
    bool
        True if travel context is detected
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in TRAVEL_KEYWORDS)


def _find_closest_city(word: str) -> Optional[str]:
    """Find the closest city name using fuzzy matching.

    Parameters
    ----------
    word : str
        The word to match

    Returns
    -------
    Optional[str]
        City name if match found, None otherwise
    """
    cities = _load_city_names()
    all_french_cities = _load_all_french_cities()

    # Normalize to lowercase for comparison, but return original city name
    word_lower = word.lower()

    # IMPORTANT: If the word is already a valid French city, don't correct it!
    # This prevents "Nanterre" from being corrected to "Nantes"
    if word_lower in all_french_cities:
        return None

    # Check manual corrections first
    if word_lower in MANUAL_CORRECTIONS:
        return MANUAL_CORRECTIONS[word_lower]

    # Use rapidfuzz to find the best match (case-insensitive)
    result = process.extractOne(
        word_lower,
        [city.lower() for city in cities],
        scorer=fuzz.ratio,
        score_cutoff=MIN_SIMILARITY_SCORE,
    )

    if result:
        # Find the original city name (with proper capitalization)
        matched_lower = result[0]
        for city in cities:
            if city.lower() == matched_lower:
                return city
    return None


def correct_city_names(text: str) -> str:
    """Correct common phonetic errors in city names using fuzzy matching.

    Only corrects if the text has travel context or location prepositions.

    Parameters
    ----------
    text : str
        The input text that may contain incorrectly transcribed city names

    Returns
    -------
    str
        The text with corrected city names
    """
    # First check if we have travel context
    has_context = _has_travel_context(text)

    # If no travel context, return original text
    if not has_context:
        return text

    corrected = text
    words = re.findall(r"\b\w+\b", text)

    # Track corrections to avoid multiple passes
    corrections = {}

    for word in words:
        # Skip very short words (less than 3 characters)
        if len(word) < 3:
            continue

        # Check if this word is close to a city name
        city_name = _find_closest_city(word)

        if city_name:
            # Only correct if it's not already the correct city name
            if word.lower() != city_name.lower():
                corrections[word] = city_name

    # Apply corrections
    for wrong, correct in corrections.items():
        # Use word boundaries to avoid partial replacements
        pattern = r"\b" + re.escape(wrong) + r"\b"
        corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)

    return corrected


def get_city_names() -> List[str]:
    """Get all available city names.

    Returns
    -------
    List[str]
        List of city names from stations.csv
    """
    return _load_city_names().copy()
