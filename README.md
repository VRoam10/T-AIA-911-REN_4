# Travel Order Resolver

Le **Travel Order Resolver** est un projet académique en Python qui transforme une phrase en français décrivant un trajet en une structure exploitable : il extrait une gare de départ et une gare d’arrivée, puis calcule le chemin le plus court entre ces deux gares à l’aide d’un graphe et de l’algorithme de Dijkstra.
This setup includes:

- **Git hooks** for conventional commits validation
- **Pre-commit hooks** for automatic code formatting and linting
- **GitHub Actions** workflow for CI/CD checks on PRs

Le projet fonctionne entièrement hors‑ligne et sert de baseline pour explorer différentes approches de traitement du langage naturel (NLP).

## Description

À partir d’une phrase simple comme :

> Je veux aller de Paris à Marseille

le pipeline actuel :

1. analyse le texte pour détecter les gares mentionnées (départ et arrivée) ;
2. charge un graphe de transport fictif à partir de fichiers CSV ;
3. calcule le plus court chemin entre les deux gares avec Dijkstra ;
4. affiche le trajet et la distance totale.

Le but est pédagogique : poser une architecture propre et extensible, puis améliorer progressivement la partie NLP.

## Architecture globale

Le code est organisé en trois briques principales :

- `src/nlp/` – traitement du texte :
  - `extract_stations.py` extrait la gare de départ et la gare d’arrivée à partir d’une phrase en français, en utilisant des règles simples basées sur les noms de villes connus dans le CSV.
- `src/graph/` – gestion du graphe :
  - `load_graph.py` charge les stations et les arêtes depuis les fichiers CSV (`stations.csv`, `edges.csv`) et construit une structure de graphe en mémoire.
  - `dijkstra.py` implémente l’algorithme de Dijkstra pour calculer le plus court chemin entre deux gares.
- `src/pipeline.py` – orchestration :
  - enchaîne l’extraction des gares, le chargement du graphe et l’appel à Dijkstra ;
  - utilise actuellement une phrase d’exemple codée en dur pour valider la baseline.

D’autres modules (`src/nlp/intent.py`, `src/io/input_text.py`) existent mais ne sont pas encore implémentés dans la baseline actuelle.

## Données

Le dossier `data/` contient les fichiers CSV utilisés par le pipeline :

- `data/stations.csv` : liste de gares fictives (identifiant de gare et nom de ville) ;
- `data/edges.csv` : liste d’arêtes entre gares avec une distance (poids).

Ces données :

- sont entièrement **fictives** ;
- servent uniquement à vérifier que le pipeline de base fonctionne ;
- pourront être remplacées plus tard par un jeu de données plus riche ou plus réaliste.

## Lancer la pipeline

### Prérequis

- Python **3.11** ou version supérieure.

Aucune dépendance externe n’est nécessaire : le projet utilise uniquement la bibliothèque standard de Python.

### Commande

Depuis la racine du projet (`T-AIA-911-REN_4`), exécuter :

```bash
python -m src.pipeline
```

Cela lance le pipeline de démonstration avec la phrase d’exemple intégrée dans `src/pipeline.py`.

## Exemple de sortie

Pour la phrase :

> Je veux aller de Paris à Marseille

une exécution typique donne :

```text
Sentence: Je veux aller de Paris à Marseille
Departure: PAR
Arrival: MAR
Shortest path: PAR -> DIJ -> LYO -> MAR
Total distance: 835.0 km
```

Les valeurs exactes dépendent uniquement des fichiers CSV fournis dans `data/`, qui sont conçus pour valider ce scénario de base.

## État actuel du projet

À ce stade :

- le **pipeline baseline** est fonctionnel (chargement du graphe, extraction de gares sur phrases simples, calcul du plus court chemin) ;
- la partie **NLP** est volontairement simple et **basée sur des règles** :
  - recherche de noms de villes connus dans la phrase ;
  - première ville trouvée = départ, deuxième ville = arrivée ;
- l’objectif est d’**améliorer progressivement le NLP** et de **comparer plusieurs approches** (règles, modèles plus avancés, etc.), tout en gardant la même interface.

Certains modules (détection d’intention, entrée texte générique, tests automatisés) sont encore à l’état de squelette et seront complétés dans les étapes suivantes du projet.

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

### flake8 (Linter)

Checks code for style issues and errors.

```bash
flake8 .
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
- Linting (flake8)
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
