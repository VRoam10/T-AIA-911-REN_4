"""Tests for station extraction from user sentences."""

from src.nlp.extract_stations import StationExtractionResult, extract_stations


def test_extract_stations_empty_sentence():
    """Empty input should return no stations and a clear error."""
    result = extract_stations("")
    assert result.departure is None
    assert result.arrival is None
    assert result.error == "Empty sentence."


def test_extract_stations_simple_order():
    sentence = "Je veux aller de Rennes à Toulouse Matabiau"
    result = extract_stations(sentence)

    assert isinstance(result, StationExtractionResult)
    # Rennes et Toulouse Matabiau existent dans stations.csv avec leurs codes FR_*.
    assert result.departure == "FR_RENNES"
    assert result.arrival == "FR_TOULOUSE_MATABIAU"
    assert result.error is None


def test_extract_stations_unknown_city():
    """Unknown cities should yield a partial result with an error."""
    sentence = "Je veux aller de Gotham à Metropolis"
    result = extract_stations(sentence)

    assert result.departure is None or result.arrival is None
    assert result.error == "Could not detect both departure and arrival stations."
