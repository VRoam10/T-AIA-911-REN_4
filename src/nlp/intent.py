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

import re
from enum import Enum, auto
from typing import List, Optional, Set

try:
    from langdetect import LangDetectException, detect
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
    non_french_chars = set("ñł¿¡ß")
    # Check for Italian-specific patterns
    if any(char in text_lower for char in non_french_chars) or "'è" in text_lower:
        return False

    # Check for very short texts first
    if len(text.strip()) < 6:  # Very short texts like "Oui", "Non", etc.
        return text_lower in {
            "oui",
            "non",
            "salut",
            "bonjour",
            "merci",
            "au revoir",
            "coucou",
            "ok",
        }

    # Check for common French expressions that might be short
    common_french = {
        "c'est parti",
        "allons-y",
        "ça va",
        "comment ça va",
        "je vais bien",
        "merci beaucoup",
        "à bientôt",
        "à plus tard",
        "à tout à l'heure",
        "hé ! ça va ?",
        "hé! ça va",
        "salut ça va",
        "salut, ça va",
        "hé ! ça va",
        "allons-y !",
        "allons-y",
        "on y va",
        "on y va !",
    }
    if text_lower in common_french:
        return True

    # Then try with langdetect for more complex cases
    if detect is not None:
        try:
            # For mixed language texts, check if there's significant French content
            if any(
                fr_word in text_lower
                for fr_word in [" est ", " et ", " dans ", " avec "]
            ):
                # If we have French structure, it's likely French
                if any(
                    fr_word in text_lower
                    for fr_word in [" la ", " le ", " les ", " un ", " une "]
                ):
                    return True

            # For short texts, combine multiple lines to improve accuracy
            if len(text) < 15:
                extended_text = f"{text} {text} {text}"
                lang = detect(extended_text)
                if lang == "fr":
                    return True
                return _basic_french_detection(text)

            # For longer texts, use direct detection
            lang = detect(text)
            if lang == "fr":
                return True
            return _basic_french_detection(text)
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
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "d'",
        "au",
        "aux",
        "à",
        "et",
        "est",
        "dans",
        "pour",
        "avec",
        "sur",
        "par",
        "sans",
        "sous",
        "chez",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "ce",
        "cet",
        "cette",
        "ces",
        "mon",
        "ton",
        "son",
        "ma",
        "ta",
        "sa",
        "mes",
        "tes",
        "ses",
        "notre",
        "votre",
        "leur",
        "nos",
        "vos",
        "leurs",
        "qui",
        "que",
        "quoi",
        "où",
        "quand",
        "comment",
        "pourquoi",
        "depuis",
        "vers",
        # Common verbs (conjugated forms)
        "suis",
        "es",
        "est",
        "sommes",
        "êtes",
        "sont",
        "ai",
        "as",
        "a",
        "avons",
        "avez",
        "ont",
        "vais",
        "vas",
        "va",
        "allons",
        "allez",
        "vont",
        "fais",
        "fait",
        "faisons",
        "faites",
        "font",
        "dis",
        "dit",
        "disons",
        "dites",
        "disent",
        "peux",
        "peut",
        "pouvons",
        "pouvez",
        "peuvent",
        "veux",
        "veut",
        "voulons",
        "voulez",
        "veulent",
        "dois",
        "doit",
        "devons",
        "devez",
        "doivent",
        # Common expressions and short words
        "oui",
        "non",
        "merci",
        "bonjour",
        "bonsoir",
        "salut",
        "au revoir",
        "s'il vous plaît",
        "s'il te plaît",
        "excusez-moi",
        "pardon",
        "bien",
        "mal",
        "bien sûr",
        "peut-être",
        "aussi",
        "toujours",
        "jamais",
        # Common nouns
        "monsieur",
        "madame",
        "mademoiselle",
        "ami",
        "personne",
        "chose",
        "maison",
        "ville",
        "rue",
        "place",
        "gare",
        "train",
        "bus",
        "métro",
        "temps",
        "jour",
        "nuit",
        "matin",
        "soir",
        "année",
        "mois",
        "semaine",
        "travail",
        "école",
        "université",
        "professeur",
        "étudiant",
        "livre",
    }

    # Check for French-specific characters
    french_chars = set("éèêëàâçîïôùûüÿœæ")
    has_french_chars = any(char in text_lower for char in french_chars)

    # Check for common French words
    words = set(re.findall(r"[\w\']+", text_lower))
    if not words:
        return False

    french_function_words = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "d'",
        "au",
        "aux",
        "à",
        "et",
        "est",
        "dans",
        "pour",
        "avec",
        "sur",
        "par",
        "sans",
        "sous",
        "chez",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "ce",
        "cet",
        "cette",
        "ces",
        "mon",
        "ton",
        "son",
        "ma",
        "ta",
        "sa",
        "mes",
        "tes",
        "ses",
        "notre",
        "votre",
        "leur",
        "nos",
        "vos",
        "leurs",
        "qui",
        "que",
        "quoi",
        "où",
        "quand",
        "comment",
        "pourquoi",
        "depuis",
        "vers",
    }

    # Count French words and check for common patterns
    french_word_count = len(words.intersection(french_indicators))

    # If there are no French-specific characters, require either a function word
    # or strong French travel terms to avoid English false positives.
    french_travel_terms = {
        "trajet",
        "itineraire",
        "itinéraire",
        "gare",
        "arrêt",
        "arret",
        "aéroport",
        "aeroport",
    }
    if not has_french_chars and not words.intersection(
        french_function_words.union(french_travel_terms)
    ):
        return False

    # Special case for very short texts (1-3 words)
    if len(words) <= 3:
        # For very short texts, require at least one French word, character, or strong travel term
        return bool(
            has_french_chars
            or french_word_count >= 1
            or words.intersection(french_function_words.union(french_travel_terms))
        )

    # For longer texts, require a minimum density of French indicators
    if len(words) >= 4:
        if french_word_count < 2:
            return False
        if (french_word_count / len(words)) < 0.3:
            return False

    min_required = max(1, len(words) // 5)  # Require at least 20% French words
    return has_french_chars or french_word_count >= min_required


def _is_travel_request(text: str) -> bool:
    """Check if the text appears to be a travel request.

    Parameters
    ----------
    text : str
        The input text to analyze

    Returns
    -------
    bool
        True if the text is a travel request, False otherwise
    """
    if not text.strip():
        return False

    text_lower = text.lower().strip()

    # Common travel-related phrases in French
    travel_phrases = [
        # Aller à [lieu]
        r"\baller\s+(?:à|a\s+|au\s+|aux\s+|en\s+|à\s+la\s+|à\s+l[\'\s]|chez\s+)[\w\s-]+",
        # Se rendre à [lieu]
        r"\b(?:me\s+|je\s+)?(?:voudrais\s+|voudrai[s|t]\s+)?(?:me\s+)?rendre\s+(?:à|a\s+|au\s+|aux\s+|en\s+|à\s+la\s+|à\s+l[\'\s]|chez\s+)[\w\s-]+",
        # Comment aller à [lieu]
        r"\b(?:comment\s+)?(?:puis\s*-?\s*je\s+)?(?:aller|me\s+rendre|me\s+diriger)\s+(?:à|a\s+|au\s+|aux\s+|en\s+|vers\s+|jusqu\'?à\s*|à\s+la\s+|à\s+l[\'\s]|chez\s+)[\w\s-]+",
        # Itinéraire pour [lieu]
        r"\b(?:donn[ée]\s*-?\s*moi\s+)?(?:l[\'\s]?\s*)?(?:itin[ée]raire|trajet|chemin|route|direction)\s+(?:pour\s+|vers\s+|jusqu\'?à\s*|en\s+direction\s+de\s*|à\s+destination\s+de\s*)?[\w\s-]+",
        # De [lieu] à [lieu]
        r"\b(?:partir\s+de|depuis|de\s+(?:la\s+)?(?:gare|station|arr[êe]t)\s+de|de\s+l[\'\s]\s*(?:gare|station|arr[êe]t)\s+de)\s+[\w\s-]+\s+(?:à\s+|vers\s+|pour\s+|en\s+direction\s+de\s*|à\s+destination\s+de\s*)[\w\s-]+(?:\s+s\'?il\s+vous\s+pla[iî]t)?[?.!]?$",
        # [Lieu] - [Lieu] ou [Lieu] -> [Lieu]
        r"\b(?:donn[ée]\s*-?\s*moi\s+)?(?:l[\'\s]?\s*)?(?:itin[ée]raire|trajet|chemin|route)\s+(?:entre\s+)?[\w\s-]+\s*[-–>]\s*[\w\s-]+",
        # Formules avec "de [lieu] à [lieu]"
        r"\b(?:de\s+[\w\s-]+\s+(?:à|vers)\s+[\w\s-]+)",
        r"\b(?:depuis\s+[\w\s-]+\s+(?:jusqu\'?à|à|vers)\s+[\w\s-]+)",
        # Autres formulations courantes
        r"\b(?:je\s+cherche\s+(?:le\s+chemin\s+pour|comment\s+aller|la\s+route\s+pour)\s+[\w\s-]+)",
        r"\b(?:quelle\s+est\s+la\s+meilleure\s+façon\s+d[\'\s]?aller\s+[àa]\s+[\w\s-]+)",
        r"\b(?:comment\s+faire\s+pour\s+rejoindre\s+[\w\s-]+)",
        r"\b(?:j[\'’]?aimerais?\s+(?:aller|me\s+rendre)\s+(?:à|a\s+|au\s+|aux\s+|en\s+)[\w\s-]+)",
        r"\b(?:je\s+cherche\s+à\s+me\s+rendre\s+[àa]\s+[\w\s-]+)",
    ]

    # Vérification des motifs de voyage
    for pattern in travel_phrases:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    # Vérification spécifique pour les formats avec tiret ou flèche
    if re.search(
        r"^\s*(?:trajet|itin[ée]raire|chemin|route|direction)\s+[\w\s-]+\s*[-–>]\s*[\w\s-]+\s*$",
        text_lower,
        re.IGNORECASE,
    ):
        return True

    # Vérification des mots-clés de voyage avec contexte
    travel_keywords = [
        # Mots forts qui indiquent clairement une demande de trajet
        "itinéraire",
        "trajet",
        "chemin",
        "route",
        "direction",
        "rejoindre",
        "destination",
        "comment aller",
        "comment se rendre",
        "comment rejoindre",
        "itinéraire pour",
        "trajet pour",
        "chemin vers",
        "route vers",
        "aimerais aller",
        "voudrais aller",
        "cherche à aller",
        "cherche à me rendre",
    ]

    # Mots qui nécessitent un contexte supplémentaire
    weak_keywords = [
        "aller",
        "se rendre",
        "se diriger",
        "partir",
        "arriver",
        "gare",
        "station",
        "arrêt",
        "aéroport",
    ]

    # Vérification des faux positifs courants
    false_positive_phrases = [
        r"\b(?:la\s+)?gare\s+est\s+",
        r"\b(?:la\s+)?station\s+est\s+",
        r"\b(?:l\'aéroport\s+est\s+)",
        r"\b(?:l\'arrêt\s+est\s+)",
        r"\b(?:je\s+suis\s+(?:à la|à l\'|dans la|dans l\'|à)\s+(?:gare|station|arrêt|aéroport))",
        r"\b(?:je\s+vais\s+à\s+la\s+(?:gare|station))",
        r"\b(?:je\s+suis\s+à\s+la\s+(?:gare|station))",
        r"\b(?:la\s+(?:gare|station|arrêt|aéroport)\s+(?:de|du|des|d\'|est|sera|serait|était|serait)\s+)",
    ]

    # Vérifier les faux positifs en premier
    for pattern in false_positive_phrases:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False

    # Vérifier la présence de mots-clés
    keyword_count = 0
    for keyword in travel_keywords:
        if keyword in text_lower:
            # Si le mot-clé est présent, on vérifie s'il est suivi d'un mot (un lieu probable)
            if re.search(r"\b" + re.escape(keyword) + r"\s+[\w\s-]+", text_lower):
                keyword_count += 2  # Poids plus important si suivi d'un mot
            else:
                keyword_count += 1

    # Si plusieurs indices de voyage sont présents ou si un motif fort est détecté
    if keyword_count >= 2:
        return True

    # Vérifier les motifs de type "de [lieu] à [lieu]"
    if re.search(
        r"\b(?:de|depuis|partir de)\s+[\w\s-]+\s+(?:à|vers|jusqu\'?à|pour)\s+[\w\s-]+",
        text_lower,
        re.IGNORECASE,
    ):
        return True

    return False


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
