"""Intent detection for travel-related user inputs.

This module defines a small set of intents relevant to the project and
declares the interface used to classify a sentence. The actual
implementation will be added later and may rely on a combination of
rule-based methods and lightweight NLP techniques.
"""

from enum import Enum, auto


class Intent(Enum):
    """High-level intent categories for a user sentence."""

    TRIP = auto()
    NOT_TRIP = auto()
    NOT_FRENCH = auto()
    UNKNOWN = auto()


def detect_intent(sentence: str) -> Intent:
    """Detect the intent of a natural-language sentence.

    Parameters
    ----------
    sentence:
        The raw text provided by the user describing a potential travel
        request or any other input.

    Returns
    -------
    Intent
        The detected intent among the supported categories.

    Notes
    -----
    The concrete detection strategy (lexical rules, pattern matching,
    or other NLP techniques) will be implemented later.
    """
    raise NotImplementedError("Intent detection is not implemented yet.")
