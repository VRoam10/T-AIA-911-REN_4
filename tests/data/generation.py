"""Legacy dataset generator (deprecated).

The project now ships committed evaluation datasets under `tests/data/`:
- eval_10k.csv
- eval_500.csv
- eval_edge_500.csv

This file is kept only to avoid breaking existing docs/scripts that may still
call it. It no longer generates any CSV.
"""


def main():
    print(
        "Dataset generation has been removed: evaluation datasets are committed "
        "under tests/data/ (eval_10k.csv, eval_500.csv, eval_edge_500.csv)."
    )


if __name__ == "__main__":
    main()
