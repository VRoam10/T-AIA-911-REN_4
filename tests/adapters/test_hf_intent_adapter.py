"""Tests for HuggingFace Intent Classifier adapter."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.domain.models import Intent


class TestHuggingFaceIntentClassifier:
    """Test suite for HuggingFaceIntentClassifier."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock transformers pipeline."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def classifier_with_mock(self, mock_pipeline):
        """Create classifier with mocked pipeline."""
        with patch(
            "src.adapters.nlp.hf_intent_adapter.HuggingFaceIntentClassifier._get_classifier"
        ) as mock_get:
            mock_get.return_value = mock_pipeline
            from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

            clf = HuggingFaceIntentClassifier()
            clf._get_classifier = mock_get
            yield clf, mock_pipeline

    def test_classify_empty_string_returns_unknown(self):
        """Empty string should return UNKNOWN without calling the model."""
        from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

        with patch.object(HuggingFaceIntentClassifier, "_get_classifier") as mock_get:
            clf = HuggingFaceIntentClassifier()
            result = clf.classify("")
            assert result == Intent.UNKNOWN
            mock_get.assert_not_called()

    def test_classify_whitespace_returns_unknown(self):
        """Whitespace-only string should return UNKNOWN."""
        from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

        with patch.object(HuggingFaceIntentClassifier, "_get_classifier") as mock_get:
            clf = HuggingFaceIntentClassifier()
            result = clf.classify("   ")
            assert result == Intent.UNKNOWN
            mock_get.assert_not_called()

    def test_classify_trip_request(self, classifier_with_mock):
        """Travel request should return TRIP."""
        clf, mock_pipeline = classifier_with_mock
        mock_pipeline.return_value = {
            "labels": ["demande de voyage", "autre demande", "texte non français"],
            "scores": [0.9, 0.05, 0.05],
        }

        result = clf.classify("Je veux aller de Paris à Lyon")
        assert result == Intent.TRIP

    def test_classify_not_trip_request(self, classifier_with_mock):
        """Non-travel French request should return NOT_TRIP."""
        clf, mock_pipeline = classifier_with_mock
        mock_pipeline.return_value = {
            "labels": ["autre demande", "demande de voyage", "texte non français"],
            "scores": [0.85, 0.1, 0.05],
        }

        result = clf.classify("Quel temps fait-il aujourd'hui?")
        assert result == Intent.NOT_TRIP

    def test_classify_non_french(self, classifier_with_mock):
        """Non-French text should return NOT_FRENCH."""
        clf, mock_pipeline = classifier_with_mock
        mock_pipeline.return_value = {
            "labels": ["texte non français", "autre demande", "demande de voyage"],
            "scores": [0.95, 0.03, 0.02],
        }

        result = clf.classify("Hello, how are you?")
        assert result == Intent.NOT_FRENCH

    def test_classify_low_confidence_returns_unknown(self, classifier_with_mock):
        """Low confidence scores should return UNKNOWN."""
        clf, mock_pipeline = classifier_with_mock
        mock_pipeline.return_value = {
            "labels": ["demande de voyage", "autre demande", "texte non français"],
            "scores": [0.35, 0.33, 0.32],
        }

        result = clf.classify("Test ambiguous text")
        assert result == Intent.UNKNOWN

    def test_confidence_threshold_customizable(self):
        """Custom confidence threshold should be respected."""
        from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "labels": ["demande de voyage", "autre demande", "texte non français"],
            "scores": [0.6, 0.3, 0.1],
        }

        with patch.object(
            HuggingFaceIntentClassifier, "_get_classifier", return_value=mock_pipeline
        ):
            # With default threshold (0.5), should return TRIP
            clf_default = HuggingFaceIntentClassifier()
            assert clf_default.classify("Test") == Intent.TRIP

            # With high threshold (0.7), should return UNKNOWN
            clf_strict = HuggingFaceIntentClassifier(confidence_threshold=0.7)
            assert clf_strict.classify("Test") == Intent.UNKNOWN

    def test_unload_clears_cache(self):
        """unload() should clear the cache."""
        from src.adapters.cache.memory_cache import InMemoryCache
        from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

        cache = InMemoryCache(name="test_intent")
        clf = HuggingFaceIntentClassifier(cache=cache)

        # Simulate cached data
        cache.set("test_key", "test_value")
        assert cache.size() == 1

        clf.unload()
        assert cache.size() == 0


class TestGetIntentClassifier:
    """Test the factory function."""

    def test_rule_based_returns_detect_intent(self):
        """rule_based strategy should return the legacy detect_intent function."""
        from src.nlp.intent import detect_intent, get_intent_classifier

        classifier = get_intent_classifier("rule_based")
        assert classifier is detect_intent

    def test_default_returns_rule_based(self):
        """Default strategy should be rule_based."""
        from src.nlp.intent import detect_intent, get_intent_classifier

        classifier = get_intent_classifier()
        assert classifier is detect_intent

    def test_unknown_strategy_returns_rule_based(self):
        """Unknown strategy should fall back to rule_based."""
        from src.nlp.intent import detect_intent, get_intent_classifier

        classifier = get_intent_classifier("nonexistent")
        assert classifier is detect_intent

    @patch("src.adapters.nlp.hf_intent_adapter.HuggingFaceIntentClassifier")
    def test_hf_xnli_returns_hf_classifier(self, mock_cls):
        """hf_xnli strategy should return HuggingFace classifier."""
        from src.nlp.intent import get_intent_classifier

        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        classifier = get_intent_classifier("hf_xnli")
        assert classifier is mock_instance.classify


@pytest.mark.skipif(
    os.environ.get("SKIP_SLOW_TESTS", "1") == "1",
    reason="Slow test: downloads HuggingFace model (~2GB)",
)
class TestHuggingFaceIntentClassifierIntegration:
    """Integration tests that require the actual model (slow)."""

    def test_real_trip_classification(self):
        """Real model should correctly classify travel requests."""
        from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

        clf = HuggingFaceIntentClassifier()
        result = clf.classify("Je veux aller de Paris à Lyon")
        assert result == Intent.TRIP

    def test_real_not_trip_classification(self):
        """Real model should correctly classify non-travel requests."""
        from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

        clf = HuggingFaceIntentClassifier()
        result = clf.classify("Quel temps fait-il?")
        assert result == Intent.NOT_TRIP

    def test_real_non_french_classification(self):
        """Real model should correctly classify non-French text."""
        from src.adapters.nlp.hf_intent_adapter import HuggingFaceIntentClassifier

        clf = HuggingFaceIntentClassifier()
        result = clf.classify("Hello, how are you?")
        assert result == Intent.NOT_FRENCH
