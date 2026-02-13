# Travel Order Resolver

The **Travel Order Resolver** is an academic project in Python that transforms a sentence in French describing a journey into a usable structure: it extracts a departure station and an arrival station, then calculates the shortest route between these two stations using a graph and Dijkstra's algorithm.
This setup includes:

- **Git hooks** for conventional commits validation
- **Pre-commit hooks** for automatic code formatting and linting
- **GitHub Actions** workflow for CI/CD checks on PRs

The project runs entirely offline and serves as a baseline for exploring different approaches to natural language processing (NLP).

## Description

Starting with a simple sentence such as:

> Je veux aller de Paris à Marseille

the current pipeline:

1. analyses the text to detect the stations mentioned (departure and arrival);
2. loads a fictitious transport graph from CSV files;
3. calculates the shortest route between the two stations using Dijkstra;
4. displays the route and total distance.

The aim is educational: to establish a clean and extensible architecture, then gradually improve the NLP part.

## Overall architecture

The code is organised into three main components:

- `src/nlp/` – text processing:
  - `extract_stations.py` extracts the departure station and arrival station from a sentence in French, using simple rules based on the names of cities known in the CSV.
- `src/graph/` – graph management:
  - `load_graph.py` loads stations and edges from CSV files (`stations.csv`, `edges.csv`) and builds a graph structure in memory.
  - `dijkstra.py` implements Dijkstra's algorithm to calculate the shortest path between two stations.
- `src/pipeline.py` – orchestration:
  - chains together station extraction, graph loading, and the Dijkstra call;
  - currently uses a hard-coded example sentence to validate the baseline.

Other modules (`src/nlp/intent.py`, `src/io/input_text.py`) exist but are not yet implemented in the current baseline.

## Data

The `data/` folder contains the CSV files used by the pipeline:

- `data/stations.csv`: list of fictitious stations (station ID and city name);
- `data/edges.csv`: list of edges between stations with a distance (weight).

This data:

- is entirely **fictitious**;
- is used solely to verify that the basic pipeline is functioning;
- may be replaced later by a richer or more realistic dataset.

## Launching the pipeline

### Prerequisites

- Python **3.11** or higher.

No external dependencies are required: the project only uses the standard Python library.

### Command

From the project root (`T-AIA-911-REN_4`), run:

```bash
python -m src.pipeline
```

This launches the demonstration pipeline with the example sentence included in `src/pipeline.py`.

## Run the Gradio App

Recommended (module mode, no PYTHONPATH tweaks needed):

```bash
python -m apps.app
```

Or use the interactive launcher:

```bash
python start.py
```

Then choose the Windows option when prompted.

## Example output

For the sentence:

> Je veux aller de Rennes à Toulouse

a typical execution gives:

```text
Sentence: Je veux aller de Rennes à Toulouse
Shortest path: FR_RENNES -> FR_BABINIERE -> FR_TOULOUSE_MATABIAU
Total distance: 562.0 km
```

The exact values depend solely on the CSV files provided in `data/`, which are designed to validate this basic scenario.

## Current status of the project

At this stade :

- The **baseline pipeline** is functional (graph loading, station extraction from simple sentences, shortest path calculation).
- The **NLP** part is deliberately simple and **rule-based**:
  - Search for known city names in the sentence.
  - first city found = departure, second city = arrival;
- the objective is to **gradually improve the NLP** and **compare several approaches** (rules, more advanced models, etc.), while keeping the same interface.

Some modules (intention detection, generic text input, automated testing) are still in the skeleton stage and will be completed in subsequent stages of the project.

## Run Tests

```bash
# Fast tests only (~10-15s, recommended for development)
pytest -n auto -m "not slow"

# Full suite including evaluations (~20 min, or ~10 min with -n 2)
pytest -n 2

# Only evaluation tests (generates PDF reports in test_results/)
pytest -m slow
```

The test suite includes:
- **Fast tests** (~88 tests): Unit and integration tests
- **Slow tests** (2 tests marked `@pytest.mark.slow`): NLP and pipeline evaluations on 10K sentences

### Generate the dataset

```bash
pytest tests/data/generation.py
```

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

Install the French spaCy model for text processing:

```bash
python -m spacy download fr_core_news_md
```

And install [cuda v12.4](https://developer.nvidia.com/cuda-12-4-1-download-archive) if you have a compatible GPU for faster-whisper

### 2. Install Pre-commit Hooks

```bash
pre-commit install --hook-type commit-msg --hook-type pre-commit
```

This installs two types of hooks:

- **pre-commit**: Runs linters/formatters before each commit
- **commit-msg**: Validates commit messages follow conventional commit format

### 3. Make Your First Commit

Use commitizen for interactive commit messages:

```bash
cz commit
```

Or write conventional commits manually:

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in module"
git commit -m "docs: update README"
```

## Conventional Commit Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `perf`: Performance improvements

**Examples:**

```
feat(auth): add login functionality
fix(api): resolve timeout issue
docs: update installation guide
```

## Linting Tools

### Black (Code Formatter)

Automatically formats your code to conform to PEP 8.

```bash
black .
```

### isort (Import Sorter)

Sorts and organizes imports.

```bash
isort .
```

### mypy (Type Checker)

Performs static type checking.

```bash
mypy .
```

### Run All Checks Manually

```bash
pre-commit run --all-files
```

## GitHub Actions Workflow

The `.github/workflows/lint.yml` workflow runs automatically on:

- Pull requests to `main` or `develop`
- Pushes to `main` or `develop`

It checks:

- Code formatting (Black)
- Import sorting (isort)
- Type checking (mypy)
- Tests (pytest, if tests exist)

## Project Structure

```
your-project/
├── .github/
│   └── workflows/
│       └── lint.yml          # CI/CD workflow
├── src/                       # Source code
├── tests/                     # Test files
├── .pre-commit-config.yaml   # Pre-commit configuration
├── pyproject.toml            # Project configuration
├── requirements.txt          # Production dependencies
└── requirements-dev.txt      # Development dependencies
```

## Tips

1. **Bypass hooks** (not recommended):

   ```bash
   git commit --no-verify -m "message"
   ```

2. **Update pre-commit hooks**:

   ```bash
   pre-commit autoupdate
   ```

3. **Skip specific files**:
   Add to `.pre-commit-config.yaml`:

   ```yaml
   exclude: ^(path/to/file\.py|another/file\.py)$
   ```

4. **Configure in IDE**:
   - VS Code: Install Python, Black, and isort extensions
   - PyCharm: Enable Black and configure in Settings → Tools

## Troubleshooting

**Hook installation failed:**

```bash
pre-commit clean
pre-commit install --hook-type commit-msg --hook-type pre-commit
```
