"""Tests for travel request detection."""

import pytest

from src.nlp.intent import Intent, _is_travel_request, detect_intent

# Test cases for travel requests
TRAVEL_REQUESTS = [
    # Basic travel requests
    "Comment aller à Paris ?",
    "Je veux aller à la gare de Lyon",
    "Donne-moi l'itinéraire pour Marseille",
    "Comment se rendre à l'aéroport Charles de Gaulle ?",
    "Je cherche le chemin pour la tour Eiffel",
    "Quelle est la route pour Bordeaux ?",
    "Direction Montpellier s'il vous plaît",
    "Je dois aller à la gare du Nord",
    "Peux-tu me montrer comment aller à Nice ?",
    "Quel est le trajet pour Toulouse ?",
    # With departure and arrival
    "De Paris à Lyon",
    "Comment aller de Paris à Marseille ?",
    "Je veux aller de la gare du Nord à la Défense",
    "Itinéraire entre Lyon et Marseille",
    "Trajet Nantes - Rennes",
    "Partir de Lille pour aller à Strasbourg",
    "Depuis Bordeaux jusqu'à Toulouse",
    "De la gare Montparnasse à la tour Eiffel",
    # Different formulations
    "Je cherche à me rendre à Lyon",
    "J'aimerais aller à Marseille",
    "Pourriez-vous m'indiquer comment aller à Nice ?",
    "Je voudrais me rendre à la gare de Lyon",
    "Quel chemin prendre pour aller à Bordeaux ?",
    "Montre-moi la direction pour aller à Toulouse",
    "Je dois me rendre à l'aéroport d'Orly",
    "Quelle est la meilleure façon d'aller à Lille ?",
    "Comment faire pour rejoindre la gare Saint-Lazare ?",
    "Je cherche comment aller au stade de France",
    "Indiquez-moi la route pour Versailles",
]

# Test cases for non-travel requests
NON_TRAVEL_REQUESTS = [
    "Bonjour, comment ça va ?",
    "Quel temps fait-il à Paris ?",
    "Je m'appelle Jean",
    "Quelle heure est-il ?",
    "Je vais bien merci",
    "C'est une belle journée",
    "Je vais au cinéma ce soir",
    "J'aime beaucoup les voyages en train",
    "La gare est fermée aujourd'hui",
    "Je préfère le bus au métro",
]


@pytest.mark.parametrize("text", TRAVEL_REQUESTS)
def test_detect_travel_requests(text):
    """Test that travel requests are correctly identified."""
    assert _is_travel_request(text), f"Failed to detect travel request: {text}"
    assert detect_intent(text) == Intent.TRIP, f"Incorrect intent for: {text}"


@pytest.mark.parametrize("text", NON_TRAVEL_REQUESTS)
def test_detect_non_travel_requests(text):
    """Test that non-travel requests are not identified as travel requests."""
    assert not _is_travel_request(text), f"Incorrectly detected as travel: {text}"
    assert detect_intent(text) != Intent.TRIP, f"Incorrect intent for: {text}"


def test_edge_cases():
    """Test edge cases and special characters."""
    # Empty string
    assert not _is_travel_request("")
    assert detect_intent("") == Intent.UNKNOWN

    # Very short text
    assert not _is_travel_request("à")
    assert not _is_travel_request("Paris")

    # With extra spaces
    assert _is_travel_request("  Comment  aller   à  Paris  ?  ")
    assert detect_intent("  Aller  à  Lyon  ") == Intent.TRIP
