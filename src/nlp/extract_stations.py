"""Station extraction from natural-language travel orders.

This module exposes a typed interface for extracting departure and
arrival stations from a user sentence. The goal is to offer a clean
contract where multiple NLP approaches can later be compared and
evaluated (e.g. simple heuristics, pattern-based methods, or more
advanced techniques).
"""

from dataclasses import dataclass
from typing import Optional


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
    raise NotImplementedError("Station extraction is not implemented yet.")

