import csv
import random
import unicodedata

TARGET_SENTENCES = 10000

ASR_ERRORS = {
    "paris": ["pari", "pariss"],
    "lyon": ["lion", "lyonn"],
    "marseille": ["marseile"],
    "gare": ["gar"],
    "saint": ["st"],
}

SLANG = ["euh", "bah", "stp", "vite", "du coup"]

NON_FRENCH = [
    "book ticket now",
    "quiero viajar mañana",
    "ich fahre nach berlin",
    "random words here",
]


def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def inject_noise(text):
    words = text.split()
    for i, w in enumerate(words):
        key = normalize(w)
        if key in ASR_ERRORS and random.random() < 0.3:
            words[i] = random.choice(ASR_ERRORS[key])
    if random.random() < 0.2:
        text = random.choice(SLANG) + " " + " ".join(words)
    else:
        text = " ".join(words)
    return text


def load_cities(path):
    cities = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cities.setdefault(r["canonical"], []).append(r["variant"])
    return cities


def load_structures(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    cities = load_cities("cities.csv")
    structures = load_structures("sentence_structures.csv")
    city_keys = list(cities.keys())

    with open("generated_sentences.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentenceID", "sentence", "expected_output"])

        for i in range(TARGET_SENTENCES):
            sid = f"{i:05d}"
            s = random.choice(structures)

            dep, arr = random.sample(city_keys, 2)
            dep_v = random.choice(cities[dep])
            arr_v = random.choice(cities[arr])

            sentence = s["template"].format(DEP=dep_v, ARR=arr_v)
            sentence = inject_noise(sentence).lower()

            if s["expects_results"] == "CORRECT":
                expected = f"CORRECT"
            elif s["expects_results"] == "NOT_TRIP":
                expected = "NOT_TRIP"
            elif s["expects_results"] == "NOT_FRENCH":
                expected = "NOT_FRENCH"
            else:
                expected = "UNKNOWN"

            writer.writerow([sid, sentence, expected])

    print(f"{TARGET_SENTENCES} sentences generated.")


if __name__ == "__main__":
    main()
