"""Fine-tune CamemBERT for NER token classification (BIO tagging).

Labels: O, B-DEP, I-DEP, B-ARR, I-ARR

Usage:
    python training/train_ner.py

Requirements:
    - Run training/data/prepare_ner_data.py first
    - GPU recommended (CUDA)

Output:
    training/models/ner-camembert/
"""

import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (
    BASE_MODEL,
    DATA_DIR,
    NER_ID2LABEL,
    NER_LABEL2ID,
    NER_LABELS,
    NER_MODEL_DIR,
    NER_TRAINING,
)

MAX_LENGTH = 128


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize and realign BIO labels to subword tokens.

    When a word is split into multiple subword tokens, the first subword
    gets the original label and subsequent subwords get -100 (ignored in loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word
                label_ids.append(labels[word_id])
            else:
                # Subsequent subword of same word -> ignore in loss
                label_ids.append(-100)
            previous_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    """Compute entity-level metrics using seqeval."""
    from seqeval.metrics import f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Convert IDs back to label strings, ignoring -100
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_clean = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_seq.append(NER_LABELS[l])
                pred_seq_clean.append(NER_LABELS[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    return {
        "overall_precision": precision_score(true_labels, pred_labels),
        "overall_recall": recall_score(true_labels, pred_labels),
        "overall_f1": f1_score(true_labels, pred_labels),
    }


def main() -> None:
    dataset_path = DATA_DIR / "ner_dataset"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run: python training/data/prepare_ner_data.py")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))
    print(
        f"  Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}"
    )

    print(f"Loading tokenizer: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Tokenizing and aligning labels...")
    tokenized = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    print(f"Loading model: {BASE_MODEL} (num_labels={len(NER_LABELS)})...")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(NER_LABELS),
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    output_dir = str(NER_MODEL_DIR)
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="none",
        fp16=True,
        **NER_TRAINING,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nTraining complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"  Test precision: {test_metrics['eval_overall_precision']:.4f}")
    print(f"  Test recall: {test_metrics['eval_overall_recall']:.4f}")
    print(f"  Test F1: {test_metrics['eval_overall_f1']:.4f}")

    # Save the best model and tokenizer
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
