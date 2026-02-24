"""NLP adapters - Implementations of NLP-related ports.

Available implementations:
- RuleBasedStationExtractor: Simple rule-based extraction using CSV lookup
- SpaCyNERAdapter: SpaCy-based NER with lazy model loading (FIXED!)
- HuggingFaceNERAdapter: HuggingFace transformers NER
- RuleBasedIntentClassifier: Rule-based intent classification
- HuggingFaceIntentClassifier: HuggingFace zero-shot intent classification
- FineTunedIntentClassifier: Fine-tuned CamemBERT intent classification
- FineTunedNERAdapter: Fine-tuned CamemBERT NER extraction
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

# HuggingFaceIntentClassifier is optional (requires transformers)
try:
    from .hf_intent_adapter import HuggingFaceIntentClassifier

    __all__.append("HuggingFaceIntentClassifier")
except ImportError:
    pass

# Fine-tuned adapters are optional (requires transformers)
try:
    from .finetuned_intent_adapter import FineTunedIntentClassifier

    __all__.append("FineTunedIntentClassifier")
except ImportError:
    pass

try:
    from .finetuned_ner_adapter import FineTunedNERAdapter

    __all__.append("FineTunedNERAdapter")
except ImportError:
    pass
