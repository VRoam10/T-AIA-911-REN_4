"""Convert generated_sentences.csv to HuggingFace BIO NER dataset.

Only CORRECT sentences (valid travel requests) are used, since they
contain identifiable departure and arrival stations.

Usage:
    python training/data/prepare_ner_data.py

Output:
    training/data/ner_dataset/ (HuggingFace DatasetDict with train/val/test splits)
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from training.config import (
    DATA_DIR,
    GENERATED_CSV,
    NER_LABELS,
    STATIONS_CSV,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from training.data.bio_annotations import annotate_bio, load_station_names


def build_ner_records(csv_path: Path, stations_csv: Path) -> list[dict]:
    """Build BIO-annotated records from CORRECT sentences.

    For each CORRECT sentence, we use the station names from the CSV
    to identify departure/arrival locations and produce BIO tags.
    """
    from src.nlp.extract_stations import _canonicalize, _load_stations

    station_lookup = _load_stations()
    station_names = load_station_names(stations_csv)

    records = []
    skipped = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["expected_output"].strip() != "CORRECT":
                continue

            sentence = row["sentence"].strip()
            if not sentence:
                skipped += 1
                continue

            # Use rule-based extraction to get departure/arrival codes
            from src.nlp.extract_stations import extract_stations

            result = extract_stations(sentence)

            if result.departure is None or result.arrival is None:
                skipped += 1
                continue

            # Find the original station/city names for the codes
            dep_name = _find_name_for_code(
                sentence, result.departure, station_lookup, station_names
            )
            arr_name = _find_name_for_code(
                sentence, result.arrival, station_lookup, station_names
            )

            if dep_name is None or arr_name is None:
                skipped += 1
                continue

            tokens, tags = annotate_bio(sentence, dep_name, arr_name)

            if not tokens:
                skipped += 1
                continue

            # Verify we got at least one B-DEP and one B-ARR
            if "B-DEP" not in tags or "B-ARR" not in tags:
                skipped += 1
                continue

            tag_ids = [NER_LABELS.index(t) for t in tags]
            records.append({"tokens": tokens, "ner_tags": tag_ids})

    print(f"  Built {len(records)} NER records, skipped {skipped}")
    return records


def _find_name_for_code(
    sentence: str,
    code: str,
    station_lookup: dict[str, str],
    station_names: dict[str, list[str]],
) -> str | None:
    """Find the original station/city name that appears in the sentence for a given code."""
    from src.nlp.extract_stations import _canonicalize

    canon_sentence = _canonicalize(sentence)

    # Collect all names that map to this code
    candidates = []
    for canon_name, station_code in station_lookup.items():
        if station_code == code:
            candidates.append(canon_name)

    # Sort by length (prefer longer matches)
    candidates.sort(key=len, reverse=True)

    # Find which candidate appears in the sentence
    for canon_name in candidates:
        if canon_name in canon_sentence:
            # Return the original (non-canonicalized) name
            if canon_name in station_names:
                return station_names[canon_name][0]
            # Fallback: return the canonicalized name itself
            return canon_name

    return None


def main() -> None:
    print(f"Loading sentences from {GENERATED_CSV}...")
    records = build_ner_records(GENERATED_CSV, STATIONS_CSV)
    print(f"  Total NER records: {len(records)}")

    if not records:
        print("ERROR: No NER records generated. Check your data.")
        sys.exit(1)

    # Show tag distribution
    from collections import Counter

    tag_counts: Counter = Counter()
    for r in records:
        for tag_id in r["ner_tags"]:
            tag_counts[NER_LABELS[tag_id]] += 1
    print("  Tag distribution:")
    for tag, count in sorted(tag_counts.items()):
        print(f"    {tag}: {count}")

    # Create HuggingFace dataset
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=NER_LABELS)),
        }
    )
    dataset = Dataset.from_list(records, features=features)

    # Split: train / val / test
    train_test = dataset.train_test_split(test_size=VAL_RATIO + TEST_RATIO, seed=42)
    val_test = train_test["test"].train_test_split(
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        seed=42,
    )

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    output_dir = DATA_DIR / "ner_dataset"
    dataset_dict.save_to_disk(str(output_dir))
    print(f"\nDataset saved to {output_dir}")
    for split, ds in dataset_dict.items():
        print(f"  {split}: {len(ds)} samples")

    # Show a few examples
    print("\nExamples:")
    for i in range(min(3, len(records))):
        tokens = records[i]["tokens"]
        tags = [NER_LABELS[t] for t in records[i]["ner_tags"]]
        print(f"\n  Sentence: {' '.join(tokens)}")
        for tok, tag in zip(tokens, tags):
            if tag != "O":
                print(f"    {tok:20s} {tag}")


if __name__ == "__main__":
    main()
