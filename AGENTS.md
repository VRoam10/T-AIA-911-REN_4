# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the core pipeline and domain logic.
  - `src/nlp/` contains text parsing helpers (e.g., station extraction).
  - `src/graph/` loads CSV data and runs Dijkstra shortest-path.
  - `src/pipeline.py` orchestrates end-to-end execution.
- `data/` contains the fictive CSV inputs (`stations.csv`, `edges.csv`).
- `tests/` contains pytest suites for language and travel detection.
- `apps/` includes CLI/app entry points (`app.py`, `app-mac.py`).
- Project config and tooling live in `pyproject.toml` and `.pre-commit-config.yaml`.

## Build, Test, and Development Commands
- Install dev tooling: `pip install -r requirements-dev.txt`
- Run the demo pipeline: `python -m src.pipeline`
- Run tests: `pytest`
- Format code: `black .`
- Sort imports: `isort .`
- Type-check: `mypy .`
- Run all hooks locally: `pre-commit run --all-files`

## Coding Style & Naming Conventions
- Python 3.11+ is required.
- Formatting is enforced by Black (88-char lines) and isort (Black profile).
- Test discovery follows pytest defaults in `pyproject.toml`:
  - Files: `test_*.py`
  - Functions: `test_*`
  - Classes: `Test*`

## Testing Guidelines
- Use pytest for all tests; add new tests under `tests/`.
- Prefer small, focused parametrized tests (see `tests/test_language_detection.py`).
- Run `pytest` before submitting changes; CI mirrors the same checks.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits (see recent history):
  - Examples: `feat: add new feature`, `fix: resolve bug`, `docs: update README`.
- Use commitizen for interactive commits: `cz commit`.
- Branch names should follow the repo pattern (type/slug), e.g. `feat/add-result`, `fix/lint`, `docs/update-guidelines`.
- PRs should include:
  - A short description of the change and impact.
  - Tests run (or a note if none were needed).
  - Linked issue or context if applicable.
  - Confirmation that CI checks pass on the PR (formatting, isort, mypy, pytest) as enforced by `.github/workflows/lint.yml` for `main`/`develop`.

## Security & Configuration Tips
- Data under `data/` is fictive and safe to modify for experiments.
- Keep model downloads (e.g., spaCy French model) local and out of version control.
