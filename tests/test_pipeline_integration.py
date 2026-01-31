"""Tests d'intégration pour le pipeline complet avec détection d'intent."""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp.intent import Intent, detect_intent
from src.pipeline import solve_travel_order


def test_pipeline_with_valid_french_trip():
    """Test le pipeline avec une demande de voyage valide en français."""
    sentence = "Je veux aller de Paris à Lyon"
    result = solve_travel_order(sentence)

    assert "Error" not in result, f"Ne devrait pas avoir d'erreur: {result}"
    assert "Shortest path" in result, f"Devrait contenir le chemin: {result}"
    assert "Total distance" in result, f"Devrait contenir la distance: {result}"
    print(f"✓ Test voyage valide: {sentence}")
    print(f"  Résultat: {result}\n")


def test_pipeline_with_non_french_input():
    """Test le pipeline avec une entrée non-française."""
    sentence = "How do I get from Paris to Lyon?"
    result = solve_travel_order(sentence)

    assert (
        result == "Error: Input is not in French"
    ), f"Devrait détecter non-français: {result}"
    print(f"✓ Test non-français: {sentence}")
    print(f"  Résultat: {result}\n")


def test_pipeline_with_french_non_trip():
    """Test le pipeline avec du français qui n'est pas un voyage."""
    sentence = "Quel temps fait-il à Paris aujourd'hui ?"
    result = solve_travel_order(sentence)

    assert (
        result == "Error: Input is not a travel request"
    ), f"Devrait détecter non-voyage: {result}"
    print(f"✓ Test français non-voyage: {sentence}")
    print(f"  Résultat: {result}\n")


def test_pipeline_with_empty_input():
    """Test le pipeline avec une entrée vide."""
    sentence = "   "
    result = solve_travel_order(sentence)

    assert (
        result == "Error: Empty or invalid input"
    ), f"Devrait détecter entrée vide: {result}"
    print(f"✓ Test entrée vide")
    print(f"  Résultat: {result}\n")


def test_pipeline_with_multiple_valid_trips():
    """Test le pipeline avec plusieurs demandes valides."""
    test_cases = [
        "Je veux aller de Rennes à Toulouse",
        "Comment aller de Marseille à Lille ?",
        "Trajet pour aller de Bordeaux à Paris",
        "Donne-moi l'itinéraire de Nice à Lyon",
    ]

    for sentence in test_cases:
        result = solve_travel_order(sentence)
        assert "Error" not in result, f"Échec pour '{sentence}': {result}"
        assert "Shortest path" in result or "No path found" in result
        print(f"✓ Test: {sentence}")
        print(f"  Résultat: {result[:60]}...\n")


def test_intent_detection_before_pipeline():
    """Test que l'intent est bien détecté avant le traitement."""
    # Vérifier que detect_intent fonctionne correctement
    test_cases = [
        ("Je veux aller de Paris à Lyon", Intent.TRIP),
        ("Quel temps fait-il ?", Intent.NOT_TRIP),
        ("Hello world", Intent.NOT_FRENCH),
        ("", Intent.UNKNOWN),
    ]

    for sentence, expected_intent in test_cases:
        intent = detect_intent(sentence)
        assert (
            intent == expected_intent
        ), f"Intent incorrect pour '{sentence}': {intent.name} != {expected_intent.name}"
        print(f"✓ Intent détecté: {sentence[:40]:<40} -> {intent.name}")

    print()


if __name__ == "__main__":
    import pytest

    # Run pytest on this file
    sys.exit(pytest.main([__file__, "-v"]))
