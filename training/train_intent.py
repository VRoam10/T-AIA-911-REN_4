"""Fine-tune CamemBERT for intent classification (TRIP / NOT_TRIP / NOT_FRENCH).

Usage:
    python training/train_intent.py

Requirements:
    - Run training/data/prepare_intent_data.py first
    - GPU recommended (CUDA)

Output:
    training/models/intent-camembert/
"""

import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (
    BASE_MODEL,
    DATA_DIR,
    INTENT_ID2LABEL,
    INTENT_LABEL2ID,
    INTENT_LABELS,
    INTENT_MODEL_DIR,
    INTENT_TRAINING,
)

MAX_LENGTH = 128


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main() -> None:
    dataset_path = DATA_DIR / "intent_dataset"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run: python training/data/prepare_intent_data.py")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))
    print(
        f"  Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}"
    )

    print(f"Loading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    print(f"Loading model: {BASE_MODEL} (num_labels={len(INTENT_LABELS)})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(INTENT_LABELS),
        id2label=INTENT_ID2LABEL,
        label2id=INTENT_LABEL2ID,
    )

    output_dir = str(INTENT_MODEL_DIR)
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="none",
        fp16=True,
        **INTENT_TRAINING,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nTraining complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"  Test accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_metrics['eval_f1']:.4f}")
    print(f"  Test F1 (weighted): {test_metrics['eval_f1_weighted']:.4f}")

    # Save the best model and tokenizer
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
