"""Tests for French language detection."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.nlp.intent import _is_french

# Test cases for French language detection
FRENCH_TEXTS = [
    # Basic French
    "Bonjour, comment allez-vous ?",
    "Je voudrais aller à Paris.",
    "Où se trouve la gare la plus proche ?",
    # With accents
    "Élémentaire mon cher Watson",
    "Hé ! Ça va ?",
    # Short texts
    "Allons-y !",
    "C'est parti",
    # With numbers and special chars
    "Rendez-vous à 14h30, c'est parfait !"
]

NON_FRENCH_TEXTS = [
    # English
    "Hello, how are you?",
    "I would like to go to Paris.",
    # Spanish
    "¿Dónde está la estación de tren?",
    # German
    "Wo ist der nächste Bahnhof?",
    # Italian
    "Dov'è la stazione?",
    # Random characters
    "asdfghjkl qwerty",
    # Empty or whitespace
    "",
    "   ",
    "\n\n"
]

@pytest.mark.parametrize("text", FRENCH_TEXTS)
def test_detect_french(text):
    """Test that French text is correctly identified."""
    assert _is_french(text), f"Failed to detect French in: {text}"

@pytest.mark.parametrize("text", NON_FRENCH_TEXTS)
def test_detect_non_french(text):
    """Test that non-French text is correctly identified."""
    assert not _is_french(text), f"Incorrectly detected as French: {text}"

def test_french_with_english_phrases():
    """Test with mixed French and English text."""
    # Should detect as French if it contains enough French
    mixed_text = "I would like to go to Paris. Où est la gare s'il vous plaît ?"
    assert _is_french(mixed_text)
    
    # Should detect as not French if not enough French
    mostly_english = "Hello, comment ça va? I'm doing great!"
    assert not _is_french(mostly_english)

def test_short_texts():
    """Test with very short texts."""
    assert _is_french("Oui")
    assert _is_french("Non")
    assert not _is_french("Yes")
    assert not _is_french("No")
    assert not _is_french("123")
    assert not _is_french("@#$")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
