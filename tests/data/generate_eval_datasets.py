"""Generate fixed evaluation datasets (10k, 500 quick, 500 edge cases)."""

from __future__ import annotations

import csv
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
STATIONS_CSV = ROOT / "data" / "stations.csv"
OUTPUT_10K = Path(__file__).resolve().parent / "eval_10k.csv"
OUTPUT_500 = Path(__file__).resolve().parent / "eval_500.csv"
OUTPUT_EDGE_500 = Path(__file__).resolve().parent / "eval_edge_500.csv"

SEED = 911


@dataclass(frozen=True)
class StationEntry:
    code: str
    city: str
    name: str


def _ascii(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def load_stations(path: Path) -> List[StationEntry]:
    entries: List[StationEntry] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("station_id") or "").strip()
            city = (row.get("city") or "").strip()
            name = (row.get("station_name") or "").strip()
            if not code or not (city or name):
                continue
            entries.append(StationEntry(code=code, city=city, name=name))
    return entries


TEMPLATES = [
    ("T01", "Je veux aller de {DEP} a {ARR}", "POSITIVE", "EASY"),
    ("T02", "Trajet de {DEP} a {ARR}", "POSITIVE", "EASY"),
    ("T03", "Comment aller de {DEP} a {ARR} ?", "POSITIVE", "EASY"),
    ("T04", "Donne moi l itineraire de {DEP} a {ARR}", "POSITIVE", "MEDIUM"),
    ("T05", "Je pars de {DEP} pour aller a {ARR}", "POSITIVE", "MEDIUM"),
    ("T06", "De {DEP} a {ARR}", "POSITIVE", "EASY"),
    ("T07", "Train de {DEP} vers {ARR}", "POSITIVE", "MEDIUM"),
    ("T08", "Itineraire entre {DEP} et {ARR}", "POSITIVE", "MEDIUM"),
    ("N01", "Quel temps fait il a {ARR} ?", "NEGATIVE", "EASY"),
    ("N02", "Meteo a {ARR} aujourd hui", "NEGATIVE", "EASY"),
    ("N03", "Je cherche un train pour {ARR}", "NEGATIVE", "MEDIUM"),
    ("N04", "Je vais voir un ami a {ARR}", "NEGATIVE", "MEDIUM"),
    ("N05", "Je voyage demain", "NEGATIVE", "EASY"),
    ("F01", "I need to go {ARR} from {DEP}", "NEGATIVE", "EASY"),
    ("F02", "How do I get from {DEP} to {ARR}?", "NEGATIVE", "EASY"),
    ("A01", "Je veux aller de {DEP} a {ARR} a {DEP}", "AMBIGUOUS", "HARD"),
    ("A02", "{DEP} {ARR}", "AMBIGUOUS", "MEDIUM"),
    ("A03", "Je pars de {DEP}", "AMBIGUOUS", "EASY"),
    ("A04", "Je vais a {ARR}", "AMBIGUOUS", "EASY"),
]


ASR_NOISE = {
    "paris": ["pari", "pariss"],
    "lyon": ["lion", "lyonn"],
    "marseille": ["marseile", "marsay"],
    "rennes": ["reine"],
    "lille": ["l ile", "lile"],
    "nice": ["nis"],
    "toulouse": ["toulouze"],
    "saint": ["st"],
    "gare": ["gar"],
}


def inject_noise(text: str, noise_type: str) -> str:
    if noise_type == "NONE":
        return text
    words = text.split()
    for i, w in enumerate(words):
        key = _ascii(w).lower()
        if key in ASR_NOISE:
            words[i] = random.choice(ASR_NOISE[key])
    noisy = " ".join(words)
    if noise_type == "LOWERCASED":
        return noisy.lower()
    if noise_type == "PUNCT_DROPPED":
        return noisy.replace("?", "").replace(",", "").replace(";", "")
    if noise_type == "SLANG":
        return f"euh {noisy}"
    return noisy


def pick_two(stations: List[StationEntry]) -> Tuple[StationEntry, StationEntry]:
    a, b = random.sample(stations, 2)
    return a, b


def label_for_template(template_id: str) -> Tuple[str, str]:
    for tid, _tpl, case_type, _diff in TEMPLATES:
        if tid == template_id:
            if tid.startswith("F"):
                return "NOT_FRENCH", "en"
            if case_type == "POSITIVE":
                return "TRIP", "fr"
            if case_type == "NEGATIVE":
                return "NOT_TRIP", "fr"
            return "NOT_TRIP", "fr"
    return "UNKNOWN", "unknown"


def render_row(
    case_id: str,
    dep: StationEntry,
    arr: StationEntry,
    template_id: str,
    noise_type: str,
    ambiguity_type: str,
) -> List[str]:
    tpl = next(t[1] for t in TEMPLATES if t[0] == template_id)
    intent_gt, language_gt = label_for_template(template_id)
    case_type = next(t[2] for t in TEMPLATES if t[0] == template_id)
    difficulty = next(t[3] for t in TEMPLATES if t[0] == template_id)
    text = tpl.format(DEP=_ascii(dep.city), ARR=_ascii(arr.city))
    text = inject_noise(text, noise_type)
    departure_gt = dep.code if intent_gt == "TRIP" else ""
    arrival_gt = arr.code if intent_gt == "TRIP" else ""
    return [
        case_id,
        text,
        intent_gt,
        departure_gt,
        arrival_gt,
        case_type,
        difficulty,
        ambiguity_type,
        template_id,
        noise_type,
        language_gt,
        "",
    ]


def generate_main(count: int, stations: List[StationEntry]) -> List[List[str]]:
    rows: List[List[str]] = []
    for i in range(count):
        case_id = f"{i + 1:05d}"
        dep, arr = pick_two(stations)
        template_id, _tpl, case_type, _diff = random.choice(TEMPLATES)
        noise_type = random.choice(["NONE", "LOWERCASED", "PUNCT_DROPPED"])
        ambiguity_type = "NONE"
        if case_type == "AMBIGUOUS":
            ambiguity_type = random.choice(
                ["MULTI_CITY", "ONE_SIDED", "SAME_CITY", "NOISE"]
            )
        rows.append(
            render_row(case_id, dep, arr, template_id, noise_type, ambiguity_type)
        )
    return rows


EDGE_CASES = [
    ("E001", "maseille st charles", "TRIP", "FR_MARSEILLE_ST_CHARLES", "", "AMBIGUOUS", "HARD", "NOISE", "EDGE", "ASR_TYPO", "fr", "ASR typo city"),
    ("E002", "l ile a paris", "NOT_TRIP", "", "", "AMBIGUOUS", "HARD", "PHONETIC", "EDGE", "NOISE", "fr", "Lille homophone"),
    ("E003", "reine vers paris", "NOT_TRIP", "", "", "AMBIGUOUS", "HARD", "PHONETIC", "EDGE", "NOISE", "fr", "Rennes homophone"),
    ("E004", "saint malo a rennes", "TRIP", "FR_ST_MALO", "FR_RENNES", "POSITIVE", "MEDIUM", "NONE", "EDGE", "LOWERCASED", "fr", "Saint name"),
    ("E005", "st etienne vers lyon", "TRIP", "FR_ST_ETIENNE_CHATEAUCREUX", "FR_LYON_PART_DIEU", "POSITIVE", "MEDIUM", "NONE", "EDGE", "LOWERCASED", "fr", "Abbrev saint"),
]


def generate_edge(stations: List[StationEntry], count: int) -> List[List[str]]:
    rows: List[List[str]] = []
    # Seeded deterministic edge list + noisy template-based edges
    rows.extend([list(r) for r in EDGE_CASES])
    while len(rows) < count:
        dep, arr = pick_two(stations)
        case_id = f"E{len(rows) + 1:03d}"
        template_id = random.choice(["A01", "A02", "A03", "A04", "T01"])
        noise_type = random.choice(["SLANG", "LOWERCASED", "PUNCT_DROPPED", "ASR_TYPO"])
        ambiguity_type = random.choice(
            ["MULTI_CITY", "ONE_SIDED", "SAME_CITY", "PHONETIC", "NOISE"]
        )
        row = render_row(case_id, dep, arr, template_id, noise_type, ambiguity_type)
        # force some language edge cases
        if random.random() < 0.2:
            row[2] = "NOT_FRENCH"
            row[10] = "en"
        rows.append(row)
    return rows[:count]


def write_csv(path: Path, rows: Iterable[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "text",
                "intent_gt",
                "departure_gt",
                "arrival_gt",
                "case_type",
                "difficulty",
                "ambiguity_type",
                "template_id",
                "noise_type",
                "language_gt",
                "notes",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    random.seed(SEED)
    stations = load_stations(STATIONS_CSV)
    if len(stations) < 2:
        raise RuntimeError("Not enough stations loaded.")

    rows_10k = generate_main(10_000, stations)
    rows_500 = generate_main(500, stations)
    rows_edge = generate_edge(stations, 500)

    write_csv(OUTPUT_10K, rows_10k)
    write_csv(OUTPUT_500, rows_500)
    write_csv(OUTPUT_EDGE_500, rows_edge)

    print(f"Wrote {len(rows_10k)} rows to {OUTPUT_10K}")
    print(f"Wrote {len(rows_500)} rows to {OUTPUT_500}")
    print(f"Wrote {len(rows_edge)} rows to {OUTPUT_EDGE_500}")


if __name__ == "__main__":
    main()
