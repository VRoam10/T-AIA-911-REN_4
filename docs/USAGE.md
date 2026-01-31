# Guide d'Utilisation - Travel Order Resolver

## Démarrage Rapide

### Installation

```bash
pip install -r requirements-dev.txt
python -m spacy download fr_core_news_md
```

### Utilisation en 3 Lignes

```python
from src.container import Container

service = Container.create_default().resolve(TravelResolverService)
result, error = service.resolve_safe("Je veux aller de Rennes à Toulouse")
print(result.path if not error else error)
```

---

## Exemples d'Utilisation

### 1. Résolution Simple

```python
from src.container import Container
from src.services import TravelResolverService

# Créer le service
container = Container.create_default()
service = container.resolve(TravelResolverService)

# Résoudre une requête
result, error = service.resolve_safe("Je veux aller de Paris à Lyon")

if error:
    print(f"Erreur: {error}")
else:
    print(f"Chemin: {' -> '.join(result.path)}")
    print(f"Distance: {result.total_distance_km} km")
```

**Sortie:**
```
Chemin: FR_PARIS_GARE_DE_LYON -> FR_MACON -> FR_LYON_PART_DIEU
Distance: 465.0 km
```

### 2. Avec Génération de Carte

```python
from pathlib import Path
from src.container import Container
from src.services import TravelResolverService

container = Container.create_default()
service = container.resolve(TravelResolverService)

route = service.resolve(
    "Je pars de Bordeaux pour aller à Marseille",
    generate_map=True,
    map_output_path=Path("itineraire.html")
)

print(f"Distance: {route.total_distance_km} km")
print("Carte sauvegardée: itineraire.html")
```

### 3. Avec Gestion d'Erreurs Complète

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

sentences = [
    "Je veux aller de Rennes à Toulouse",  # OK
    "Hello, how are you?",                  # NOT_FRENCH
    "Bonjour, quel temps fait-il ?",        # NOT_TRIP
    "De XYZ à ABC",                         # Stations inconnues
]

for sentence in sentences:
    print(f"\n>>> {sentence}")
    try:
        route = service.resolve(sentence)
        print(f"✓ Route: {len(route.path)} arrêts, {route.total_distance_km} km")

    except IntentClassificationError as e:
        print(f"✗ Intent: {e.detected_intent}")

    except ExtractionError as e:
        print(f"✗ Extraction: {e.message}")

    except NoRouteFoundError as e:
        print(f"✗ Pas de route: {e.departure} -> {e.arrival}")
```

**Sortie:**
```
>>> Je veux aller de Rennes à Toulouse
✓ Route: 13 arrêts, 592.0 km

>>> Hello, how are you?
✗ Intent: NOT_FRENCH

>>> Bonjour, quel temps fait-il ?
✗ Intent: NOT_TRIP

>>> De XYZ à ABC
✗ Extraction: Could not extract both departure and arrival
```

### 4. Changer la Stratégie NLP

```python
from src.config import AppConfig, NLPConfig
from src.container import Container
from src.services import TravelResolverService

# Configuration avec HuggingFace NER
config = AppConfig(
    nlp=NLPConfig(default_strategy="hf_ner")  # Options: rule_based, hf_ner, spacy
)

container = Container.create_default(config=config)
service = container.resolve(TravelResolverService)

result, _ = service.resolve_safe("Je pars de Nice pour aller à Strasbourg")
```

### 5. Utiliser les Composants Individuellement

```python
from src.container import Container
from src.ports.nlp import IntentClassifierPort, StationExtractorPort
from src.ports.graph import GraphRepositoryPort, RouteSolverPort

container = Container.create_default()

# Intent Classification
classifier = container.resolve(IntentClassifierPort)
intent = classifier.classify("Je veux aller à Paris")
print(f"Intent: {intent}")  # Intent.TRIP

# Station Extraction
extractor = container.resolve(StationExtractorPort)
extraction = extractor.extract("De Rennes à Nantes")
print(f"Départ: {extraction.departure}")   # FR_RENNES
print(f"Arrivée: {extraction.arrival}")    # FR_NANTES

# Graph Loading
repo = container.resolve(GraphRepositoryPort)
graph = repo.load()
print(f"Stations: {len(graph)}")  # 500

# Route Solving
solver = container.resolve(RouteSolverPort)
route = solver.solve(graph, "FR_RENNES", "FR_NANTES")
print(f"Distance: {route.total_distance_km} km")
```

### 6. API Legacy (Rétrocompatible)

```python
from src.pipeline import solve_travel_order

# Simple
message = solve_travel_order("Je veux aller de Rennes à Toulouse")
print(message)

# Avec options
message = solve_travel_order(
    "De Paris à Lyon",
    nlp_name="rule_based",
    generate_map=True,
    map_output_html="carte.html"
)
```

---

## Interface Gradio

### Lancer l'Interface Web

```bash
# Méthode recommandée
python -m apps.app

# Ou via launcher
python start.py
# Puis choisir l'option Windows
```

### Fonctionnalités de l'Interface

1. **Saisie texte**: Entrer directement une requête
2. **Reconnaissance vocale**: Parler votre requête (Whisper)
3. **Choix du modèle ASR**: small, medium, large-v3
4. **Choix de la stratégie NLP**: rule_based, legacy_spacy, hf_ner

---

## Configuration Rapide

### Via Variables d'Environnement

```bash
# Changer la stratégie NLP
export TOR_NLP_DEFAULT_STRATEGY=hf_ner

# Forcer CPU pour ASR
export TOR_ASR_DEVICE=cpu

# Changer le dossier des données
export TOR_GRAPH_DATA_DIR=/mon/dossier/data
```

### Via Code

```python
import os
os.environ["TOR_NLP_DEFAULT_STRATEGY"] = "hf_ner"

from src.config import reset_config
reset_config()  # Recharge la config

from src.container import Container
container = Container.create_default()
```

---

## Phrases de Test

Voici des phrases valides pour tester le système :

```python
phrases_valides = [
    "Je veux aller de Paris à Lyon",
    "Comment aller de Rennes à Nantes ?",
    "De Bordeaux à Marseille s'il vous plaît",
    "Je cherche un itinéraire de Lille à Strasbourg",
    "Trajet Toulouse - Nice",
    "Je voudrais me rendre de Montpellier à Grenoble",
]

phrases_invalides = [
    "",                              # UNKNOWN
    "Hello world",                   # NOT_FRENCH
    "Bonjour, comment ça va ?",      # NOT_TRIP
    "Quel temps fait-il à Paris ?",  # NOT_TRIP
]
```

---

## Dépannage Rapide

| Problème | Solution |
|----------|----------|
| "SpaCy model not found" | `python -m spacy download fr_core_news_md` |
| "CUDA out of memory" | `export TOR_ASR_DEVICE=cpu` |
| "No path found" | Vérifier que les villes existent dans `data/stations.csv` |
| Tests pollués | Ajouter `reset_config()` et `reset_container()` dans les fixtures |
