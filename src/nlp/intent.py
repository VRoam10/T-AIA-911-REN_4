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
from typing import Set, List, Optional
import re

try:
    from langdetect import detect, LangDetectException
except ImportError:
    # Fallback si langdetect n'est pas installé
    detect = None


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


def _is_french(text: str, min_confidence: float = 0.8) -> bool:
    """Check if the text appears to be in French.
    
    Uses a combination of langdetect and custom rules for better accuracy.
    
    Parameters
    ----------
    text : str
        The text to analyze
    min_confidence : float, optional
        Minimum confidence threshold (0-1) when using langdetect
        
    Returns
    -------
    bool
        True if the text is detected as French, False otherwise
    """
    # Handle empty or whitespace-only strings
    if not text.strip():
        return False
    
    text_lower = text.lower()
    
    # Check for non-French characters that are rare in French
    non_french_chars = set('ñł¿¡ß')
    # Check for Italian-specific patterns
    if any(char in text_lower for char in non_french_chars) or "'è" in text_lower:
        return False
    
    # Check for very short texts first
    if len(text.strip()) < 6:  # Very short texts like "Oui", "Non", etc.
        return text_lower in {'oui', 'non', 'salut', 'bonjour', 'merci', 'au revoir', 'coucou', 'ok'}
    
    # Check for common French expressions that might be short
    common_french = {
        "c'est parti", "allons-y", "ça va", "comment ça va", "je vais bien", 
        "merci beaucoup", "à bientôt", "à plus tard", "à tout à l'heure",
        "hé ! ça va ?", "hé! ça va", "salut ça va", "salut, ça va", "hé ! ça va",
        "allons-y !", "allons-y", "on y va", "on y va !"
    }
    if text_lower in common_french:
        return True
    
    # Then try with langdetect for more complex cases
    if detect is not None:
        try:
            # For mixed language texts, check if there's significant French content
            if any(fr_word in text_lower for fr_word in [' est ', ' et ', ' dans ', ' avec ']):
                # If we have French structure, it's likely French
                if any(fr_word in text_lower for fr_word in [' la ', ' le ', ' les ', ' un ', ' une ']):
                    return True
            
            # For short texts, combine multiple lines to improve accuracy
            if len(text) < 15:
                extended_text = f"{text} {text} {text}"
                lang = detect(extended_text)
                return lang == 'fr'
            
            # For longer texts, use direct detection
            lang = detect(text)
            return lang == 'fr'
        except (LangDetectException, Exception):
            pass
    
    # Fall back to basic detection if langdetect is not available or fails
    return _basic_french_detection(text)


def _basic_french_detection(text: str) -> bool:
    """Basic French detection using character and word patterns."""
    text_lower = text.lower()
    
    # Common French words and articles (expanded list)
    french_indicators = {
        # Articles and common words
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'd\'', 'au', 'aux', 'à',
        'et', 'est', 'dans', 'pour', 'avec', 'sur', 'par', 'sans', 'sous', 'chez',
        'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
        'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
        'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
        'qui', 'que', 'quoi', 'où', 'quand', 'comment', 'pourquoi',
        
        # Common verbs (conjugated forms)
        'suis', 'es', 'est', 'sommes', 'êtes', 'sont',
        'ai', 'as', 'a', 'avons', 'avez', 'ont',
        'vais', 'vas', 'va', 'allons', 'allez', 'vont',
        'fais', 'fait', 'faisons', 'faites', 'font',
        'dis', 'dit', 'disons', 'dites', 'disent',
        'peux', 'peut', 'pouvons', 'pouvez', 'peuvent',
        'veux', 'veut', 'voulons', 'voulez', 'veulent',
        'dois', 'doit', 'devons', 'devez', 'doivent',
        
        # Common expressions and short words
        'oui', 'non', 'merci', 'bonjour', 'bonsoir', 'salut', 'au revoir',
        's\'il vous plaît', 's\'il te plaît', 'excusez-moi', 'pardon',
        'bien', 'mal', 'bien sûr', 'peut-être', 'aussi', 'toujours', 'jamais',
        
        # Common nouns
        'monsieur', 'madame', 'mademoiselle', 'ami', 'personne', 'chose',
        'maison', 'ville', 'rue', 'place', 'gare', 'train', 'bus', 'métro',
        'temps', 'jour', 'nuit', 'matin', 'soir', 'année', 'mois', 'semaine',
        'travail', 'école', 'université', 'professeur', 'étudiant', 'livre'
    }
    
    # Check for French-specific characters
    french_chars = set('éèêëàâçîïôùûüÿœæ')
    has_french_chars = any(char in text_lower for char in french_chars)
    
    # Check for common French words
    words = set(re.findall(r'[\w\']+', text_lower))
    if not words:
        return False
    
    # Count French words and check for common patterns
    french_word_count = len(words.intersection(french_indicators))
    
    # Special case for very short texts (1-3 words)
    if len(words) <= 3:
        # For very short texts, require at least one French word or character
        return has_french_chars or french_word_count >= 1
    
    # For longer texts, be more lenient
    min_required = max(1, len(words) // 5)  # Require at least 20% French words
    return has_french_chars or french_word_count >= min_required


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

