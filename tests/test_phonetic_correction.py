"""Tests for phonetic correction module."""

import pytest
from src.nlp.phonetic_correction import correct_city_names, add_correction


def test_correct_reine_to_rennes_with_travel_context():
    """Test correction of 'Reine' to 'Rennes' in travel context."""
    text = "Je veux aller de Reine à Lion"
    result = correct_city_names(text)
    assert result == "Je veux aller de Rennes à Lyon"


def test_correct_with_preposition():
    """Test correction when location prepositions are present."""
    text = "Je suis à Reine"
    result = correct_city_names(text)
    assert result == "Je suis à Rennes"


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
        ("Direction Marseye", "Direction Marseille"),
        ("Vers Toulouze", "Vers Toulouse"),
        ("De Bordo à Niece", "De Bordeaux à Nice"),
        ("Aller à Lil", "Aller à Lille"),
        ("Depuis Nante", "Depuis Nantes"),
    ]

    for input_text, expected in test_cases:
        result = correct_city_names(input_text)
        assert result == expected, f"Failed for '{input_text}'"


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


def test_add_custom_correction():
    """Test adding a custom correction."""
    add_correction("marse", "Marseille")
    text = "Je vais à Marse"
    result = correct_city_names(text)
    assert result == "Je vais à Marseille"


def test_empty_string():
    """Test correction with empty string."""
    result = correct_city_names("")
    assert result == ""


def test_no_cities_in_text():
    """Test text without any city names."""
    text = "Bonjour, comment allez-vous ?"
    result = correct_city_names(text)
    assert result == text
