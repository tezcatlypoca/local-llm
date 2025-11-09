# Local LLM API

API Flask pour gérer et utiliser des modèles LLM sur GPUs AMD via PyTorch/ROCm.

## Installation

1. Créer un environnement virtuel Python :
```bash
python -m venv venv
```

2. Activer l'environnement :
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Télécharger un modèle de test

Pour tester l'API, vous pouvez télécharger un petit modèle LLM :

```bash
python scripts/download_test_model.py
```

Le script vous propose plusieurs modèles :
- **GPT2** (~500 MB) - Très rapide, parfait pour les tests
- **TinyLlama** (~2 GB) - Modèle conversationnel
- **Phi-2** (~5 GB) - Modèle Microsoft performant

Ou téléchargez directement un modèle via Python :

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Télécharge automatiquement dans ~/.cache/huggingface/hub
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

## Lancer l'API

```bash
python src/main.py
```

L'API sera accessible sur `http://localhost:5000`

## Routes disponibles

- `GET /` - Route root (statut de l'API)
- `GET /models` - Liste les modèles disponibles en local

