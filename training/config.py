"""Training configuration for fine-tuned models."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_CSV = PROJECT_ROOT / "tests" / "data" / "generated_sentences.csv"
STATIONS_CSV = PROJECT_ROOT / "data" / "stations.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Base model
BASE_MODEL = "camembert-base"

# Intent model
INTENT_MODEL_DIR = MODELS_DIR / "intent-camembert"
INTENT_LABELS = ["TRIP", "NOT_TRIP", "NOT_FRENCH"]
INTENT_LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
INTENT_ID2LABEL = {i: label for i, label in enumerate(INTENT_LABELS)}

# NER model
NER_MODEL_DIR = MODELS_DIR / "ner-camembert"
NER_LABELS = ["O", "B-DEP", "I-DEP", "B-ARR", "I-ARR"]
NER_LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: label for i, label in enumerate(NER_LABELS)}

# Training hyperparameters
INTENT_TRAINING = {
    "learning_rate": 2e-5,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "seed": 42,
}

NER_TRAINING = {
    "learning_rate": 3e-5,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "overall_f1",
    "seed": 42,
}

# Data split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# CSV expected_output -> Intent label mapping
OUTPUT_TO_INTENT = {
    "CORRECT": "TRIP",
    "NOT_TRIP": "NOT_TRIP",
    "NOT_FRENCH": "NOT_FRENCH",
}
