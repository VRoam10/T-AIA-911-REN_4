from src.nlp.extract_stations import StationExtractionResult, extract_stations


def test_extract_stations_empty_sentence():
    result = extract_stations("")
    assert result.departure is None
    assert result.arrival is None
    assert result.error == "Empty sentence."


def test_extract_stations_simple_order():
    sentence = "Je veux aller de Rennes à Toulouse"
    result = extract_stations(sentence)

    assert isinstance(result, StationExtractionResult)
    # Rennes and Toulouse exist in stations.csv with codes REN and TLS.
    assert result.departure == "REN"
    assert result.arrival == "TLS"
    assert result.error is None


def test_extract_stations_unknown_city():
    sentence = "Je veux aller de Gotham à Metropolis"
    result = extract_stations(sentence)

    assert result.departure is None or result.arrival is None
    assert result.error == "Could not detect both departure and arrival stations."

