"""NLP adapters - Implementations of NLP-related ports.

Available implementations:
- RuleBasedStationExtractor: Simple rule-based extraction using CSV lookup
- SpaCyNERAdapter: SpaCy-based NER with lazy model loading (FIXED!)
- HuggingFaceNERAdapter: HuggingFace transformers NER
- RuleBasedIntentClassifier: Intent classification
"""

from .intent_adapter import RuleBasedIntentClassifier
from .rule_based import RuleBasedStationExtractor
from .spacy_adapter import SpaCyNERAdapter

__all__ = [
    "RuleBasedStationExtractor",
    "SpaCyNERAdapter",
    "RuleBasedIntentClassifier",
]

# HuggingFaceNERAdapter is optional (requires transformers)
try:
    from .hf_ner_adapter import HuggingFaceNERAdapter

    __all__.append("HuggingFaceNERAdapter")
except ImportError:
    pass
