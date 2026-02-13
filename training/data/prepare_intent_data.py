"""Convert generated_sentences.csv to HuggingFace intent classification dataset.

Usage:
    python training/data/prepare_intent_data.py

Output:
    training/data/intent_dataset/ (HuggingFace DatasetDict with train/val/test splits)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from training.config import (
    DATA_DIR,
    GENERATED_CSV,
    INTENT_LABELS,
    OUTPUT_TO_INTENT,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def load_sentences(csv_path: Path) -> list[dict]:
    """Load and convert CSV to intent-labeled records."""
    records = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = row["expected_output"].strip()
            intent_label = OUTPUT_TO_INTENT.get(expected)
            if intent_label is None:
                continue
            records.append(
                {
                    "text": row["sentence"].strip(),
                    "label": INTENT_LABELS.index(intent_label),
                }
            )
    return records


def main() -> None:
    print(f"Loading sentences from {GENERATED_CSV}...")
    records = load_sentences(GENERATED_CSV)
    print(f"  Total records: {len(records)}")

    # Show distribution
    from collections import Counter

    dist = Counter(INTENT_LABELS[r["label"]] for r in records)
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count} ({count / len(records) * 100:.1f}%)")

    # Create HuggingFace dataset
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=INTENT_LABELS),
        }
    )
    dataset = Dataset.from_list(records, features=features)

    # Split: train / val / test
    train_test = dataset.train_test_split(
        test_size=VAL_RATIO + TEST_RATIO, seed=42, stratify_by_column="label"
    )
    val_test = train_test["test"].train_test_split(
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        seed=42,
        stratify_by_column="label",
    )

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    output_dir = DATA_DIR / "intent_dataset"
    dataset_dict.save_to_disk(str(output_dir))
    print(f"\nDataset saved to {output_dir}")
    for split, ds in dataset_dict.items():
        print(f"  {split}: {len(ds)} samples")


if __name__ == "__main__":
    main()
