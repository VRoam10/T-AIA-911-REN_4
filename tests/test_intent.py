"""Tests for the intent detection module."""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp.intent import Intent, detect_intent


def test_intent_detection():
    """Test the intent detection with various inputs."""
    # Test cases: (input_text, expected_intent).
    test_cases = [
        # French travel requests
        ("Comment aller de Paris à Lyon ?", Intent.TRIP),
        ("Je veux me rendre à Marseille", Intent.TRIP),
        ("Trajet pour aller à Bordeaux", Intent.TRIP),
        ("Depuis Toulouse vers Montpellier", Intent.TRIP),
        ("Donne-moi l'itinéraire pour Nice", Intent.TRIP),
        # French non-travel requests
        ("Quel temps fait-il à Paris ?", Intent.NOT_TRIP),
        ("Je vais bien merci", Intent.NOT_TRIP),
        ("C'est une belle journée", Intent.NOT_TRIP),
        # Non-French text
        ("How do I get to the train station?", Intent.NOT_FRENCH),
        ("Hola, ¿cómo estás?", Intent.NOT_FRENCH),
        ("こんにちは", Intent.NOT_FRENCH),
        # Edge cases
        ("", Intent.UNKNOWN),
        ("   ", Intent.UNKNOWN),
    ]

    # Run the cases and collect a readable summary for debugging.
    total = len(test_cases)
    passed = 0

    for i, (text, expected) in enumerate(test_cases, 1):
        result = detect_intent(text)
        try:
            assert result == expected, f"Expected {expected.name} for input: {text!r}"
            status = "✓"
            passed += 1
        except AssertionError as e:
            status = "✗"
            print(f"\nAssertionError: {e}")

        print(f"Test {i}/{total} {status} - Input: {text!r}")
        print(f"  Expected: {expected.name}")
        print(f"  Got:      {result.name}")
        print()

    # Print summary for easier diagnosis when running this file directly.
    success = passed == total
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    assert success, f"{total - passed} tests failed"


if __name__ == "__main__":
    success = test_intent_detection()
    sys.exit(0 if success else 1)
