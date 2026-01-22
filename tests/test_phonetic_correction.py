"""Tests for phonetic correction module with rapidfuzz."""

import pytest
from src.nlp.phonetic_correction import correct_city_names, get_city_names


def test_correct_reine_to_rennes_with_travel_context():
    """Test correction of 'Reine' to 'Rennes' in travel context."""
    text = "Je veux aller de Reine à Lion"
    result = correct_city_names(text)
    assert result == "Je veux aller de Rennes à Lyon"


def test_correct_with_travel_keyword():
    """Test correction when travel keywords are present."""
    text = "Direction Reine"
    result = correct_city_names(text)
    assert result == "Direction Rennes"


def test_no_correction_without_context():
    """Test that 'reine' is not corrected without travel context."""
    text = "La reine de France est élégante"
    result = correct_city_names(text)
    assert result == "La reine de France est élégante"


def test_correct_multiple_cities():
    """Test correction of multiple city names."""
    text = "Trajet de Reine à Lion puis à Pari"
    result = correct_city_names(text)
    assert result == "Trajet de Rennes à Lyon puis à Paris"


def test_correct_lion_to_lyon():
    """Test correction of 'Lion' to 'Lyon'."""
    text = "Comment aller à Lion ?"
    result = correct_city_names(text)
    assert result == "Comment aller à Lyon ?"


def test_correct_various_cities():
    """Test correction of various city name errors."""
    test_cases = [
        ("Direction Marseye", "Marseille"),  # Should correct
        ("Vers Toulouze", "Toulouse"),  # Should correct
        ("Aller à Lil", "Lille"),  # Should correct
        ("Depuis Nante", "Nantes"),  # Should correct
        ("Comment aller à Niece", "Nice"),  # Should correct
    ]

    for input_text, expected_city in test_cases:
        result = correct_city_names(input_text)
        assert (
            expected_city in result
        ), f"Failed for '{input_text}': expected {expected_city} in result"


def test_case_insensitive_correction():
    """Test that correction works regardless of case."""
    text = "Je veux aller de REINE à lion"
    result = correct_city_names(text)
    assert "Rennes" in result and "Lyon" in result


def test_no_correction_on_partial_words():
    """Test that partial words are not corrected."""
    text = "Je vais à la reinette"
    result = correct_city_names(text)
    assert result == "Je vais à la reinette"


def test_get_city_names():
    """Test that city names are loaded from CSV."""
    cities = get_city_names()
    assert len(cities) > 0
    assert "Paris" in cities
    assert "Lyon" in cities
    assert "Rennes" in cities


def test_empty_string():
    """Test correction with empty string."""
    result = correct_city_names("")
    assert result == ""


def test_no_cities_in_text():
    """Test text without any city names."""
    text = "Bonjour, comment allez-vous ?"
    result = correct_city_names(text)
    assert result == text
