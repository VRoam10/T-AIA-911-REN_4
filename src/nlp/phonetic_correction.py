"""Phonetic correction for speech-to-text errors with context awareness.

This module corrects common transcription errors from Whisper STT
when users say city names that are transcribed incorrectly due to
phonetic similarity. It uses context to avoid false corrections.

Example
-------
    >>> correct_city_names("Je veux aller de Reine à Lion")
    "Je veux aller de Rennes à Lyon"
    >>> correct_city_names("La reine de France")
    "La reine de France"  # No correction (wrong context)
"""

import re
from typing import Dict

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

# Mapping of phonetic errors to correct city names
PHONETIC_CORRECTIONS: Dict[str, str] = {
    # Rennes variations
    "reine": "Rennes",
    "rêne": "Rennes",
    "rènes": "Rennes",
    "rênes": "Rennes",
    # Lyon variations
    "lion": "Lyon",
    "lions": "Lyon",
    # Paris variations
    "pari": "Paris",
    # Marseille variations
    "marseye": "Marseille",
    "marsey": "Marseille",
    # Toulouse variations
    "toulouze": "Toulouse",
    # Bordeaux variations
    "bordo": "Bordeaux",
    # Nice variations
    "niece": "Nice",
    "nisse": "Nice",
    # Lille variations
    "lil": "Lille",
    # Nantes variations
    "nante": "Nantes",
    # Strasbourg variations
    "strasbour": "Strasbourg",
    # Montpellier variations
    "montpelier": "Montpellier",
    "montpelié": "Montpellier",
    # Dijon variations
    "djon": "Dijon",
    # Reims variations
    "reim": "Reims",
    "rains": "Reims",
}


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


def _has_location_preposition_before(text: str, word_pos: int) -> bool:
    """Check if there's a location preposition before the word.

    Parameters
    ----------
    text : str
        The full text
    word_pos : int
        Position of the word in the text

    Returns
    -------
    bool
        True if a location preposition is found nearby
    """
    # Get text before the word (up to 20 characters back)
    before_text = text[max(0, word_pos - 20) : word_pos].lower()

    return any(prep in before_text for prep in LOCATION_PREPOSITIONS)


def correct_city_names(text: str) -> str:
    """Correct common phonetic errors in city names using context.

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

    corrected = text

    # Apply corrections with context awareness
    for wrong, correct in PHONETIC_CORRECTIONS.items():
        # Find all occurrences of the wrong word
        pattern = r"\b" + re.escape(wrong) + r"\b"

        # Process matches in reverse order to maintain positions
        matches = list(re.finditer(pattern, corrected, flags=re.IGNORECASE))
        for match in reversed(matches):
            word_pos = match.start()

            # Decide if we should correct based on context
            should_correct = has_context or _has_location_preposition_before(
                corrected, word_pos
            )

            if should_correct:
                # Replace this occurrence
                corrected = corrected[:word_pos] + correct + corrected[match.end() :]

    return corrected


def add_correction(wrong: str, correct: str) -> None:
    """Add a new phonetic correction to the mapping.

    Parameters
    ----------
    wrong : str
        The incorrect transcription
    correct : str
        The correct city name
    """
    PHONETIC_CORRECTIONS[wrong.lower()] = correct


def get_corrections() -> Dict[str, str]:
    """Get all available phonetic corrections.

    Returns
    -------
    Dict[str, str]
        Dictionary of incorrect -> correct mappings
    """
    return PHONETIC_CORRECTIONS.copy()
