# Documentation Technique - Travel Order Resolver

## Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Flux de Données](#flux-de-données)
4. [Guide d'Utilisation](#guide-dutilisation)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Extension du Système](#extension-du-système)
8. [Tests](#tests)
9. [Dépannage](#dépannage)

---

## Vue d'Ensemble

### Objectif

Le **Travel Order Resolver** est un système de traitement de requêtes de voyage en français. Il transforme une phrase naturelle comme :

> "Je veux aller de Rennes à Toulouse"

En un itinéraire optimisé avec :
- Stations de départ et d'arrivée identifiées
- Chemin le plus court calculé (algorithme de Dijkstra)
- Distance totale en kilomètres
- Carte interactive optionnelle

### Pipeline de Traitement

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Entrée    │───▶│   Intent    │───▶│ Extraction  │───▶│   Route     │
│   Texte     │    │ Detection   │    │  Stations   │    │  Finding    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │
                         ▼                  ▼                  ▼
                   Est-ce du         Quelles sont      Quel est le
                   français ?        les stations ?    chemin optimal ?
                   Est-ce un
                   voyage ?
```

---

## Architecture

### Architecture Hexagonale (Ports & Adapters)

Le projet suit une architecture hexagonale qui sépare :

- **Domain** : Modèles métier et erreurs (aucune dépendance externe)
- **Ports** : Interfaces abstraites (Protocols Python)
- **Adapters** : Implémentations concrètes des ports
- **Services** : Orchestration des cas d'utilisation

```
                    ┌─────────────────────────────────────┐
                    │           APPLICATION               │
                    │  ┌─────────────────────────────┐   │
                    │  │         SERVICES            │   │
                    │  │  TravelResolverService      │   │
                    │  │  ExtractionService          │   │
                    │  └─────────────────────────────┘   │
                    │              │                      │
                    │  ┌───────────┴───────────┐         │
                    │  │        DOMAIN         │         │
                    │  │  models.py, errors.py │         │
                    │  └───────────────────────┘         │
                    └─────────────┬───────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
     ┌──────┴──────┐       ┌──────┴──────┐       ┌──────┴──────┐
     │    PORTS    │       │    PORTS    │       │    PORTS    │
     │  (Inbound)  │       │  (Outbound) │       │  (Outbound) │
     └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
            │                     │                     │
     ┌──────┴──────┐       ┌──────┴──────┐       ┌──────┴──────┐
     │  ADAPTERS   │       │  ADAPTERS   │       │  ADAPTERS   │
     │   (CLI,     │       │   (NLP,     │       │   (Graph,   │
     │   Gradio)   │       │   ASR)      │       │   Geocode)  │
     └─────────────┘       └─────────────┘       └─────────────┘
```

### Structure des Dossiers

```
src/
├── config.py               # Configuration centralisée (Pydantic Settings)
├── container.py            # Conteneur d'injection de dépendances
│
├── domain/                 # Couche Domaine (0 dépendances)
│   ├── __init__.py
│   ├── models.py          # Modèles immutables (frozen dataclasses)
│   └── errors.py          # Erreurs typées du domaine
│
├── ports/                  # Interfaces abstraites (Protocols)
│   ├── __init__.py
│   ├── nlp.py             # StationExtractorPort, IntentClassifierPort
│   ├── graph.py           # GraphRepositoryPort, RouteSolverPort
│   ├── geocoding.py       # GeocoderPort
│   ├── asr.py             # ASRModelPort
│   ├── rendering.py       # MapRendererPort
│   └── cache.py           # CachePort
│
├── adapters/               # Implémentations concrètes
│   ├── __init__.py
│   ├── nlp/
│   │   ├── rule_based.py          # Extraction par règles (CSV lookup)
│   │   ├── spacy_adapter.py       # SpaCy NER (lazy loading!)
│   │   ├── hf_ner_adapter.py      # HuggingFace NER
│   │   └── intent_adapter.py      # Classification d'intent
│   ├── graph/
│   │   ├── csv_repository.py      # Chargement CSV du graphe
│   │   └── dijkstra_solver.py     # Algorithme Dijkstra
│   ├── geocoding/
│   │   └── nominatim_adapter.py   # Geocoding OpenStreetMap
│   ├── asr/
│   │   └── whisper_adapter.py     # Faster-Whisper ASR
│   ├── rendering/
│   │   └── folium_adapter.py      # Cartes interactives
│   └── cache/
│       ├── memory_cache.py        # Cache en mémoire (thread-safe)
│       └── null_cache.py          # Cache null (pour tests)
│
├── services/               # Orchestration des cas d'utilisation
│   ├── __init__.py
│   ├── travel_resolver.py # Service principal
│   └── extraction_service.py  # Service d'extraction NLP
│
└── [legacy modules]        # Modules historiques (toujours fonctionnels)
    ├── pipeline.py
    ├── nlp/
    ├── graph/
    └── viz/
```

---

## Flux de Données

### 1. Classification d'Intent

```python
sentence = "Je veux aller de Paris à Lyon"
         │
         ▼
┌─────────────────────────────────┐
│    IntentClassifierPort         │
│    (RuleBasedIntentClassifier)  │
└─────────────────────────────────┘
         │
         ▼
    Intent.TRIP  ✓
```

**Intents possibles :**
- `Intent.TRIP` - Requête de voyage valide en français
- `Intent.NOT_TRIP` - Français mais pas une requête de voyage
- `Intent.NOT_FRENCH` - Pas en français
- `Intent.UNKNOWN` - Entrée vide ou invalide

### 2. Extraction des Stations

```python
sentence = "Je veux aller de Paris à Lyon"
         │
         ▼
┌─────────────────────────────────┐
│    StationExtractorPort         │
│    (RuleBasedStationExtractor   │
│     ou SpaCyNERAdapter          │
│     ou HuggingFaceNERAdapter)   │
└─────────────────────────────────┘
         │
         ▼
StationExtractionResult(
    departure="FR_PARIS_GARE_DE_LYON",
    arrival="FR_LYON_PART_DIEU",
    raw_locations=("Paris", "Lyon"),
    confidence=1.0
)
```

### 3. Calcul de Route

```python
departure = "FR_PARIS_GARE_DE_LYON"
arrival = "FR_LYON_PART_DIEU"
         │
         ▼
┌─────────────────────────────────┐
│    GraphRepositoryPort          │  ◀── Charge le graphe (CSV)
│    (CSVGraphRepository)         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    RouteSolverPort              │  ◀── Dijkstra's algorithm
│    (DijkstraRouteSolver)        │
└─────────────────────────────────┘
         │
         ▼
RouteResult(
    path=("FR_PARIS_GARE_DE_LYON", ..., "FR_LYON_PART_DIEU"),
    total_distance_km=465.0,
    stations=(Station(...), ...)
)
```

---

## Guide d'Utilisation

### Installation

```bash
# Cloner le projet
git clone <repo-url>
cd T-AIA-911-REN_4

# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements-dev.txt

# Installer le modèle SpaCy français
python -m spacy download fr_core_news_md

# Installer les hooks pre-commit
pre-commit install --hook-type commit-msg --hook-type pre-commit
```

### Utilisation Basique (Nouvelle Architecture)

```python
from src.container import Container
from src.services import TravelResolverService

# Créer le conteneur avec toutes les dépendances
container = Container.create_default()

# Résoudre le service principal
service = container.resolve(TravelResolverService)

# Utiliser le service
result, error = service.resolve_safe("Je veux aller de Rennes à Toulouse")

if error:
    print(f"Erreur: {error}")
else:
    print(f"Chemin: {' -> '.join(result.path)}")
    print(f"Distance: {result.total_distance_km} km")
```

### Utilisation avec Gestion d'Erreurs

```python
from src.container import Container
from src.services import TravelResolverService
from src.domain.errors import (
    IntentClassificationError,
    ExtractionError,
    NoRouteFoundError,
)

container = Container.create_default()
service = container.resolve(TravelResolverService)

try:
    route = service.resolve("Je veux aller de Paris à Marseille")
    print(f"Route trouvée: {route.path}")
    print(f"Distance: {route.total_distance_km} km")

except IntentClassificationError as e:
    print(f"Intent invalide: {e.message}")
    print(f"Intent détecté: {e.detected_intent}")

except ExtractionError as e:
    print(f"Extraction échouée: {e.message}")
    print(f"Stratégie utilisée: {e.strategy_used}")

except NoRouteFoundError as e:
    print(f"Pas de route: {e.departure} -> {e.arrival}")
```

### Utilisation avec Génération de Carte

```python
from pathlib import Path
from src.container import Container
from src.services import TravelResolverService

container = Container.create_default()
service = container.resolve(TravelResolverService)

route = service.resolve(
    "Je veux aller de Rennes à Toulouse",
    generate_map=True,
    map_output_path=Path("./ma_carte.html")
)

print(f"Carte générée: ma_carte.html")
```

### Utilisation Legacy (Toujours Supportée)

```python
from src.pipeline import solve_travel_order

# Simple utilisation
message = solve_travel_order("Je veux aller de Rennes à Toulouse")
print(message)

# Avec options
message = solve_travel_order(
    "Je veux aller de Paris à Lyon",
    nlp_name="rule_based",  # ou "hf_ner", "legacy_spacy"
    path_name="dijkstra",
    generate_map=True,
    map_output_html="trajectory.html"
)
```

### Interface Gradio

```bash
# Lancer l'interface web
python -m apps.app

# Ou via le launcher interactif
python start.py
```

---

## Configuration

### Variables d'Environnement

Toutes les configurations sont centralisées via Pydantic Settings et peuvent être surchargées par des variables d'environnement :

```bash
# NLP
TOR_NLP_DEFAULT_STRATEGY=rule_based  # Options: rule_based, hf_ner, spacy
TOR_NLP_HF_NER_MODEL=Jean-Baptiste/camembert-ner
TOR_NLP_SPACY_MODEL=fr_core_news_md

# ASR (Speech-to-Text)
TOR_ASR_DEVICE=auto          # Options: auto, cuda, cpu
TOR_ASR_COMPUTE_TYPE=float16
TOR_ASR_DEFAULT_MODEL=large-v3
TOR_ASR_FALLBACK_DEVICE=cpu
TOR_ASR_FALLBACK_COMPUTE_TYPE=int8

# Graphe
TOR_GRAPH_DATA_DIR=/path/to/data
TOR_GRAPH_STATIONS_FILE=stations.csv
TOR_GRAPH_EDGES_FILE=edges.csv

# Geocoding
TOR_GEO_USER_AGENT=travel-order-resolver
TOR_GEO_TIMEOUT_SECONDS=10
TOR_GEO_RATE_LIMIT_DELAY=1.0
TOR_GEO_MAX_RETRIES=2

# Logging
TOR_LOG_LEVEL=INFO
TOR_LOG_STRUCTURED=false
```

### Configuration Programmatique

```python
from src.config import AppConfig, NLPConfig, get_config, reset_config

# Obtenir la configuration globale
config = get_config()
print(config.nlp.default_strategy)  # "rule_based"
print(config.graph.stations_path)    # Path to stations.csv

# Créer une configuration personnalisée
custom_config = AppConfig(
    nlp=NLPConfig(default_strategy="hf_ner"),
)

# Utiliser avec le container
from src.container import Container
container = Container.create_default(config=custom_config)
```

---

## API Reference

### Domain Models

#### `StationExtractionResult`

```python
@dataclass(frozen=True, slots=True)
class StationExtractionResult:
    departure: Optional[str] = None      # Code station départ
    arrival: Optional[str] = None        # Code station arrivée
    raw_locations: tuple[str, ...] = ()  # Locations brutes extraites
    confidence: float = 1.0              # Score de confiance [0-1]
    error: Optional[str] = None          # Message d'erreur si échec

    @property
    def is_complete(self) -> bool: ...   # True si départ ET arrivée trouvés

    @property
    def is_success(self) -> bool: ...    # True si pas d'erreur et complet
```

#### `RouteResult`

```python
@dataclass(frozen=True, slots=True)
class RouteResult:
    path: tuple[str, ...]                # Séquence de codes station
    total_distance_km: float             # Distance totale
    stations: tuple[Station, ...] = ()   # Détails des stations

    @property
    def is_empty(self) -> bool: ...      # True si pas de chemin

    @property
    def num_stops(self) -> int: ...      # Nombre d'arrêts
```

#### `Intent`

```python
class Intent(Enum):
    TRIP = auto()        # Requête de voyage valide
    NOT_TRIP = auto()    # Français, mais pas un voyage
    NOT_FRENCH = auto()  # Pas en français
    UNKNOWN = auto()     # Entrée invalide
```

### Services

#### `TravelResolverService`

```python
class TravelResolverService:
    def resolve(
        self,
        sentence: str,
        generate_map: bool = False,
        map_output_path: Optional[Path] = None,
    ) -> RouteResult:
        """
        Résout une requête de voyage.

        Raises:
            IntentClassificationError: Si l'intent n'est pas un voyage
            ExtractionError: Si l'extraction des stations échoue
            NoRouteFoundError: Si aucun chemin n'existe
        """
        ...

    def resolve_safe(
        self,
        sentence: str,
        generate_map: bool = False,
        map_output_path: Optional[Path] = None,
    ) -> tuple[Optional[RouteResult], Optional[str]]:
        """
        Comme resolve(), mais retourne (None, error_message) au lieu de raise.
        """
        ...

    def format_result(
        self,
        route: RouteResult,
        map_path: Optional[Path] = None
    ) -> str:
        """
        Formate le résultat en string lisible.
        """
        ...
```

### Ports (Interfaces)

#### `StationExtractorPort`

```python
class StationExtractorPort(Protocol):
    def extract(self, sentence: str) -> StationExtractionResult:
        """Extrait les stations de départ et d'arrivée."""
        ...
```

#### `IntentClassifierPort`

```python
class IntentClassifierPort(Protocol):
    def classify(self, sentence: str) -> Intent:
        """Classifie l'intent d'une phrase."""
        ...
```

#### `CachePort`

```python
class CachePort(Protocol[T]):
    def get(self, key: str) -> Optional[T]: ...
    def set(self, key: str, value: T) -> None: ...
    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T: ...
    def clear(self) -> int: ...
    def invalidate(self, key: str) -> bool: ...
    def size(self) -> int: ...
```

---

## Extension du Système

### Ajouter une Nouvelle Stratégie NLP

1. **Créer l'adapter** dans `src/adapters/nlp/` :

```python
# src/adapters/nlp/my_custom_extractor.py
from dataclasses import dataclass
from src.domain.models import StationExtractionResult

@dataclass
class MyCustomExtractor:
    def extract(self, sentence: str) -> StationExtractionResult:
        # Votre logique d'extraction
        departure = self._find_departure(sentence)
        arrival = self._find_arrival(sentence)

        return StationExtractionResult(
            departure=departure,
            arrival=arrival,
            raw_locations=tuple(self._extract_locations(sentence)),
        )
```

2. **Enregistrer dans le container** :

```python
# Dans src/container.py, méthode create_default()
from .adapters.nlp.my_custom_extractor import MyCustomExtractor

def create_station_extractor() -> StationExtractorPort:
    strategy = config.nlp.default_strategy
    if strategy == "my_custom":
        return MyCustomExtractor()
    # ... autres stratégies
```

### Ajouter un Nouveau Solver de Route

```python
# src/adapters/graph/astar_solver.py
from dataclasses import dataclass
from src.domain.models import RouteResult
from src.ports.graph import Graph

@dataclass
class AStarRouteSolver:
    def solve(self, graph: Graph, departure: str, arrival: str) -> RouteResult:
        # Implémenter A* ici
        path, distance = self._astar(graph, departure, arrival)
        return RouteResult(path=tuple(path), total_distance_km=distance)
```

### Créer un Adapter de Cache Personnalisé

```python
# src/adapters/cache/redis_cache.py
from dataclasses import dataclass
from typing import Optional, Callable, TypeVar
import redis

T = TypeVar("T")

@dataclass
class RedisCache:
    host: str = "localhost"
    port: int = 6379

    def __post_init__(self):
        self._client = redis.Redis(host=self.host, port=self.port)

    def get(self, key: str) -> Optional[T]:
        value = self._client.get(key)
        return pickle.loads(value) if value else None

    def set(self, key: str, value: T) -> None:
        self._client.set(key, pickle.dumps(value))

    # ... autres méthodes
```

---

## Tests

### Exécuter les Tests

```bash
# Tous les tests
pytest

# Avec couverture
pytest --cov=src --cov-report=term-missing

# Tests spécifiques
pytest tests/test_graph.py -v
pytest tests/test_intent.py -v

# Exclure les tests lents
pytest --ignore=tests/test_strategies_evaluation.py
```

### Écrire des Tests avec le Container

```python
import pytest
from src.container import Container
from src.adapters.cache import NullCache
from src.ports.cache import CachePort

@pytest.fixture
def test_container():
    """Container de test avec cache null."""
    container = Container()
    container.register(CachePort, lambda: NullCache())
    # Ajouter d'autres mocks si nécessaire
    return container

def test_travel_resolution(test_container):
    # Configurer le container de test
    from src.adapters.nlp import RuleBasedStationExtractor
    from src.ports.nlp import StationExtractorPort

    test_container.register(
        StationExtractorPort,
        lambda: RuleBasedStationExtractor()
    )

    extractor = test_container.resolve(StationExtractorPort)
    result = extractor.extract("Je vais de Paris à Lyon")

    assert result.departure is not None
    assert result.arrival is not None
```

### Nettoyer les Caches entre Tests

```python
import pytest
from src.config import reset_config
from src.container import reset_container

@pytest.fixture(autouse=True)
def cleanup():
    """Nettoie l'état global entre chaque test."""
    yield
    reset_config()
    reset_container()
```

---

## Dépannage

### Problèmes Courants

#### "SpaCy model not found"

```bash
# Solution: installer le modèle
python -m spacy download fr_core_news_md
```

#### "CUDA out of memory"

```bash
# Solution: utiliser le CPU
export TOR_ASR_DEVICE=cpu
export TOR_ASR_COMPUTE_TYPE=int8
```

#### "Geocoding timeout"

```bash
# Solution: augmenter le timeout
export TOR_GEO_TIMEOUT_SECONDS=30
```

#### Tests échouent avec des caches pollués

```python
# Dans conftest.py
@pytest.fixture(autouse=True)
def reset_caches():
    from src.container import reset_container
    from src.config import reset_config
    yield
    reset_container()
    reset_config()
```

### Logging de Debug

```python
import logging

# Activer le logging détaillé
logging.basicConfig(level=logging.DEBUG)

# Ou via variable d'environnement
# export TOR_LOG_LEVEL=DEBUG
```

### Vérifier la Configuration

```python
from src.config import get_config

config = get_config()
print(f"NLP Strategy: {config.nlp.default_strategy}")
print(f"ASR Device: {config.asr.device}")
print(f"Graph Data: {config.graph.data_dir}")
```

---

## Changelog Architecture

### v2.0.0 - Architecture Hexagonale

- **Ajouté**: Couche Domain avec modèles immutables
- **Ajouté**: Ports (interfaces Protocol)
- **Ajouté**: Adapters pour tous les composants
- **Ajouté**: Container d'injection de dépendances
- **Ajouté**: Configuration centralisée (Pydantic Settings)
- **Corrigé**: SpaCy chargé à l'import → lazy loading
- **Corrigé**: Caches globaux sans invalidation → CachePort injectable
- **Corrigé**: Swallowing silencieux d'exceptions → erreurs typées + logs
- **Corrigé**: Bug cache ASR (clé != device réel après fallback)

### v1.0.0 - Version Initiale

- Pipeline monolithique fonctionnel
- Stratégies NLP (rule-based, SpaCy, HuggingFace)
- Dijkstra pour le pathfinding
- Interface Gradio
