# Training Fine-tuned Models

This directory contains scripts to fine-tune CamemBERT models for:
1. **Intent classification** (TRIP / NOT_TRIP / NOT_FRENCH)
2. **NER slot filling** with BIO tagging (B-DEP, I-DEP, B-ARR, I-ARR, O)

## Prerequisites

```bash
pip install -r requirements-dev.txt
```

GPU (CUDA) is recommended for training.

## Quick Start

### 1. Prepare Datasets

```bash
# Intent dataset (from generated_sentences.csv)
python training/data/prepare_intent_data.py

# NER dataset (BIO annotations from CORRECT sentences)
python training/data/prepare_ner_data.py
```

Output:
- `training/data/intent_dataset/` - HuggingFace DatasetDict (train/val/test)
- `training/data/ner_dataset/` - HuggingFace DatasetDict (train/val/test)

### 2. Train Models

```bash
# Fine-tune intent classifier
python training/train_intent.py

# Fine-tune NER model
python training/train_ner.py
```

Output:
- `training/models/intent-camembert/` - Fine-tuned intent model
- `training/models/ner-camembert/` - Fine-tuned NER model

### 3. Evaluate

Run the evaluation tests to compare with other strategies:

```bash
pytest -m slow -s
```

The fine-tuned models will automatically appear in the evaluation if the model directories exist.

## Configuration

### Training Hyperparameters

Edit `training/config.py` to adjust:

| Parameter | Intent | NER |
|-----------|--------|-----|
| Base model | camembert-base | camembert-base |
| Learning rate | 2e-5 | 3e-5 |
| Epochs | 5 | 10 |
| Batch size | 16 | 16 |
| Warmup ratio | 0.1 | 0.1 |
| Early stopping | patience=2 | patience=3 |

### Using Fine-tuned Models at Runtime

Set environment variables:

```bash
# Use fine-tuned NER for station extraction
TOR_NLP_DEFAULT_STRATEGY=finetuned_ner

# Use fine-tuned intent classifier
TOR_NLP_INTENT_STRATEGY=finetuned_intent

# Custom model paths (optional)
TOR_NLP_FINETUNED_NER_MODEL=training/models/ner-camembert
TOR_NLP_FINETUNED_INTENT_MODEL=training/models/intent-camembert
```

Or in Python:

```python
from src.container import Container
from src.config import AppConfig, NLPConfig

config = AppConfig(nlp=NLPConfig(
    default_strategy="finetuned_ner",
    intent_strategy="finetuned_intent",
))
container = Container.create_default(config=config)
```

## File Structure

```
training/
├── config.py                      # Hyperparameters, labels, paths
├── data/
│   ├── bio_annotations.py        # BIO annotation helpers
│   ├── prepare_intent_data.py    # CSV -> intent dataset
│   └── prepare_ner_data.py       # CSV -> NER BIO dataset
├── train_intent.py                # CamemBERT intent fine-tuning
├── train_ner.py                   # CamemBERT NER fine-tuning
├── models/                        # Output models (gitignored)
│   ├── intent-camembert/
│   └── ner-camembert/
└── README.md                      # This file
```

## Data Format

### Intent Dataset

```json
{"text": "Je veux aller de Paris a Lyon", "label": 0}
```

Labels: `{0: "TRIP", 1: "NOT_TRIP", 2: "NOT_FRENCH"}`

### NER Dataset

```json
{"tokens": ["Je", "veux", "aller", "de", "Paris", "a", "Lyon"], "ner_tags": [0, 0, 0, 0, 1, 0, 3]}
```

Labels: `{0: "O", 1: "B-DEP", 2: "I-DEP", 3: "B-ARR", 4: "I-ARR"}`

## Architecture Integration

The fine-tuned models integrate into the hexagonal architecture:

- `src/adapters/nlp/finetuned_intent_adapter.py` implements `IntentClassifierPort`
- `src/adapters/nlp/finetuned_ner_adapter.py` implements `StationExtractorPort`
- Both use lazy loading and caching, following the same pattern as existing adapters
