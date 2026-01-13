"""Intent detection for travel-related user inputs.

This module implements a rule-based intent classification system for French travel
requests. It can distinguish between travel-related sentences and other types of
inputs, as well as detect non-French text.

Example
-------
    >>> detect_intent("Comment aller de Paris à Lyon ?")
    <Intent.TRIP: 1>
    >>> detect_intent("Quel temps fait-il à Paris ?")
    <Intent.NOT_TRIP: 2>
    >>> detect_intent("Hello, how are you?")
    <Intent.NOT_FRENCH: 3>
">>> detect_intent("")
    <Intent.UNKNOWN: 4>
"""

from enum import Enum, auto
from typing import Set, List
import re


class Intent(Enum):
    """High-level intent categories for a user sentence.
    
    Attributes
    ----------
    TRIP
        The input is a valid travel request in French.
    NOT_TRIP
        The input is in French but not a travel request.
    NOT_FRENCH
        The input is not in French.
    UNKNOWN
        The input is empty or contains only whitespace.
    """

    TRIP = auto()
    NOT_TRIP = auto()
    NOT_FRENCH = auto()
    UNKNOWN = auto()


def _is_french(text: str) -> bool:
    """Check if the text appears to be in French.
    
    This is a simple implementation that looks for common French characters and words.
    For a production system, consider using a proper language detection library.
    """
    if not text.strip():
        return False
    
    # Check for French-specific characters
    french_chars = set('éèêëàâçîïôùûüÿœæ')
    if any(char in text.lower() for char in french_chars):
        return True
    
    # Common French words and articles
    french_indicators = {
        # Articles
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'd\'', 'au', 'aux',
        # Common words
        'et', 'est', 'dans', 'pour', 'avec', 'sur', 'par', 'sans', 'sous', 'chez',
        # Pronouns
        'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
        # Common verbs
        'aller', 'être', 'avoir', 'faire', 'dire', 'voir', 'pouvoir', 'vouloir',
        'manger', 'boire', 'prendre', 'donner', 'parler', 'savoir', 'venir',
        # Common prepositions
        'à', 'de', 'dans', 'sur', 'sous', 'avec', 'pour', 'par', 'chez', 'vers'
    }
    
    words = set(re.findall(r'[\w\']+', text.lower()))
    french_word_count = len(words.intersection(french_indicators))
    
    # If we find at least 1 French word for short texts, or 2 for longer ones
    min_required = 1 if len(words) < 5 else 2
    return french_word_count >= min_required


def _is_travel_request(text: str) -> bool:
    """Check if the text appears to be a travel request."""
    if not text.strip():
        return False
    
    # Common travel-related phrases in French
    travel_phrases = [
        r'\baller\b',
        r'\b(?:trajet|itinéraire|voyage|déplacement|direction|chemin)\b',
        r'\b(?:comment|comment aller|comment se rendre|comment rejoindre)\b',
        r'\b(?:de\s+.+?\s+(?:à|vers|pour|a)\s+.+?)\b',
        r'\b(?:depuis|de|du|de la|des?)\s+.+?\s+(?:à|vers|pour|a)\s*.+',
        r'\b(?:je\s+(?:veux|voudrais|dois|cherche|souhaite)|il\s+faut)\s+(?:aller|me\s+rendre|trouver)\b',
        r'\b(?:donne|montre|trouve|cherche|indique)\s+(?:moi\s+)?(?:le\s+)?(?:trajet|itinéraire|chemin)\b',
        r'\b(?:partir|se\s+rendre|voyager)\s+(?:de\s+.+?\s+)?(?:à|vers|pour\s+)?',
        r'\b(?:direction|vers)\s+.+?\s+(?:depuis|de\s+la?\s*|des?\s*).*',
        r'\b(?:itinéraire|chemin)\s+(?:pour|vers|depuis|de\s+.+?\s+à\s+.+?)\b'
    ]
    
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in travel_phrases)


def detect_intent(sentence: str) -> Intent:
    """Classify the intent of a French natural language sentence.

    The function first checks if the input is empty or contains only whitespace,
    then verifies if the text is in French, and finally determines if it's a
    travel-related request.

    Parameters
    ----------
    sentence : str
        The input text to analyze. Can be any string, including empty strings
        or strings containing only whitespace.

    Returns
    -------
    Intent
        One of the following intent categories:
        - TRIP: The input is a valid travel request in French
        - NOT_TRIP: The input is in French but not a travel request
        - NOT_FRENCH: The input is not in French
        - UNKNOWN: The input is empty or contains only whitespace

    Examples
    --------
    >>> detect_intent("Comment aller de Paris à Lyon ?")
    <Intent.TRIP: 1>
    >>> detect_intent("Bonjour, comment ça va ?")
    <Intent.NOT_TRIP: 2>
    >>> detect_intent("Hello, how are you?")
    <Intent.NOT_FRENCH: 3>
    >>> detect_intent("    ")
    <Intent.UNKNOWN: 4>
    """
    if not sentence or not sentence.strip():
        return Intent.UNKNOWN
        
    # First check if the text is in French
    if not _is_french(sentence):
        return Intent.NOT_FRENCH
    
    # Check if it's a travel request
    if _is_travel_request(sentence):
        return Intent.TRIP
    
    # If it's French but not a travel request
    return Intent.NOT_TRIP

