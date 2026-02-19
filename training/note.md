# Fine-tuning CamemBERT pour la classification d’intention

## 🎯 Objectif

Fine-tuner `camembert-base` pour classifier des intentions :

- `TRIP`
- `NOT_TRIP`
- `NOT_FRENCH`

Script principal :

```bash
python training/train_intent.py
```

---

# 🧩 Problèmes rencontrés & Résolution

## 1️⃣ Erreur FP16 (Mixed Precision)

### ❌ Erreur initiale

```
ValueError: FP16 Mixed precision training ... can only be used on CUDA...
```

### 🔍 Cause

Le script forçait :

```python
fp16=True
```

Mais PyTorch installé était **CPU-only**, donc CUDA indisponible.

Vérification :

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Résultat :

```
torch 2.10.0+cpu
cuda? False
```

---

## 2️⃣ GPU détecté mais non utilisé

### ✅ Vérification GPU

```bash
nvidia-smi
```

Résultat :

- RTX 5070 Laptop GPU
- Driver 591.86
- CUDA 13.1

Le GPU fonctionnait parfaitement.

### ❗ Problème

PyTorch installé = version CPU-only.

---

## 3️⃣ Installation correcte de PyTorch CUDA

### 🔧 Solution

Désinstallation :

```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
```

Réinstallation CUDA 12.8 compatible RTX 50xx :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### ✅ Vérification

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Résultat :

```
torch 2.10.0+cu128
cuda? True
cuda ver 12.8
gpu NVIDIA GeForce RTX 5070 Laptop GPU
```

🎉 CUDA fonctionne correctement.

---

## 4️⃣ Nouvelle erreur : `dispatch_batches`

### ❌ Erreur

```
TypeError: Accelerator.__init__() got an unexpected keyword argument 'dispatch_batches'
```

### 🔍 Cause

Incompatibilité de versions :

- `transformers==4.38.2`
- `accelerate==1.12.0`

Transformers 4.38.2 attend une ancienne API d’Accelerate.

---

## 5️⃣ Correction compatibilité Transformers / Accelerate

### ✅ Solution retenue : downgrade accelerate

```bash
pip uninstall -y accelerate
pip install accelerate==0.28.0
```

Versions compatibles :

```bash
transformers 4.38.2
accelerate 0.28.0
```

---

# ✅ Bonnes pratiques ajoutées au script

Pour éviter les crashs CPU :

```python
import torch

use_cuda = torch.cuda.is_available()

training_args = TrainingArguments(
    ...
    fp16=use_cuda,
    ...
)

print(f"Device: {'cuda' if use_cuda else 'cpu'} | fp16={use_cuda}")
```

---

# 🔥 Configuration finale fonctionnelle

### GPU

- RTX 5070 Laptop
- Driver 591.86
- CUDA runtime 12.8 (PyTorch)
- CUDA driver 13.1

### Python

- Python 3.11.9
- venv isolé

### Packages stables

```txt
torch==2.10.0+cu128
transformers==4.38.2
accelerate==0.28.0
datasets
scikit-learn
```

---

# 🚀 Lancement de l'entraînement

```bash
python training/train_intent.py
```

Surveillance GPU :

```bash
nvidia-smi -l 1
```

---

# 📌 Points clés à retenir

- Avoir un GPU ≠ avoir PyTorch CUDA
- Vérifier `torch.cuda.is_available()`
- RTX 50xx → utiliser cu128 minimum
- Toujours matcher `transformers` et `accelerate`
- Ne jamais forcer `fp16=True` sans check CUDA

---

# 🧠 Lessons Learned

1. Les erreurs FP16 sont presque toujours liées à CUDA non détecté.
2. Les conflits `transformers` / `accelerate` sont fréquents.
3. Les RTX 50xx demandent des wheels CUDA récents.
4. Toujours vérifier les versions exactes installées dans le venv actif.
