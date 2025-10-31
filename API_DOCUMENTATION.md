# Documentation de l'API Local LLM

API Flask pour gérer et utiliser des modèles LLM sur GPUs AMD via PyTorch/ROCm.

## Base URL

- **Local** : `http://localhost:5000`
- **Réseau local** : `http://<IP_LBB>:5000` (où `<IP_LBB>` est l'adresse IP de la machine LBB)

---

## Routes disponibles

### 1. Route Root

**GET** `/`

Route de base pour vérifier que l'API est active.

**Réponse :**
```json
{
  "message": "API Flask active",
  "status": "ok"
}
```

---

### 2. Liste des modèles disponibles

**GET** `/models`

Liste tous les modèles LLM disponibles localement sur la machine.

**Paramètres :** Aucun

**Réponse :**
```json
{
  "status": "success",
  "count": 2,
  "cache_dir": "~/.cache/huggingface/hub",
  "cache_exists": true,
  "models": [
    {
      "identifier": "gpt2",
      "size_mb": 523.45,
      "revision": "abc123...",
      "path": "/path/to/model",
      "files": ["config.json", "pytorch_model.bin", ...],
      "model_type": "gpt2",
      "architectures": ["GPT2LMHeadModel"]
    }
  ]
}
```

**Codes de statut :**
- `200` : Succès
- `500` : Erreur serveur

---

### 3. Charger un modèle sur GPU

**POST** `/models/load/<model_name>`

Charge un modèle LLM sur un GPU libre.

**Paramètres URL :**
- `model_name` (path) : Nom/identifiant du modèle Hugging Face (ex: `gpt2`, `microsoft/phi-2`)

**Body JSON (optionnel) :**
```json
{
  "model_kwargs": {
    "torch_dtype": "float16"
  }
}
```

**Réponse succès (200) :**
```json
{
  "status": "success",
  "message": "Modèle \"gpt2\" chargé avec succès sur GPU 0",
  "gpu_id": 0,
  "gpu_identifier": "GPU-0",
  "model_name": "gpt2",
  "access_token": "token_securise_unique_...",
  "note": "Conservez ce token pour décharger le modèle plus tard",
  "gpu_status": {...}
}
```

**Réponse erreur (404) - Modèle introuvable :**
```json
{
  "status": "error",
  "message": "Le modèle \"inexistant\" n'existe pas localement ou sur Hugging Face Hub."
}
```

**Réponse erreur (503) - Aucun GPU libre :**
```json
{
  "status": "error",
  "message": "Aucun GPU libre. Tous les GPUs ont déjà un modèle chargé.",
  "loaded_models": [
    "GPU 0: gpt2",
    "GPU 1: microsoft/phi-2"
  ]
}
```

**Codes de statut :**
- `200` : Modèle chargé avec succès
- `404` : Modèle introuvable
- `503` : Aucun GPU libre
- `500` : Erreur lors du chargement

**⚠️ Important :** Conservez le `access_token` retourné, il est nécessaire pour décharger le modèle plus tard.

---

### 4. Décharger un modèle

**POST** `/models/unload/<gpu_id>`

Décharge un modèle d'un GPU spécifique.

**Paramètres URL :**
- `gpu_id` (int) : Numéro du GPU (0 ou 1)

**Body JSON requis :**
```json
{
  "access_token": "token_reçu_lors_du_chargement"
}
```

**Réponse succès (200) :**
```json
{
  "status": "success",
  "message": "Modèle 'gpt2' déchargé avec succès du GPU 0.",
  "gpu_id": 0,
  "gpu_identifier": "GPU-0"
}
```

**Réponse erreur (400) - Token manquant :**
```json
{
  "status": "error",
  "message": "Token d'accès requis. Fournissez le token reçu lors du chargement du modèle dans le body JSON (access_token)."
}
```

**Réponse erreur (403) - Token invalide :**
```json
{
  "status": "error",
  "message": "Token d'accès invalide. Vous n'avez pas les permissions pour décharger ce modèle.",
  "gpu_id": 0
}
```

**Réponse erreur (404) - Aucun modèle :**
```json
{
  "status": "error",
  "message": "Aucun modèle chargé sur GPU 0.",
  "gpu_id": 0
}
```

**Codes de statut :**
- `200` : Modèle déchargé avec succès
- `400` : Token manquant ou GPU ID invalide
- `403` : Token invalide (accès refusé)
- `404` : Aucun modèle sur ce GPU
- `500` : Erreur lors du déchargement

**🔒 Sécurité :** Seul le détenteur du `access_token` peut décharger le modèle, protégeant ainsi les modèles chargés par d'autres utilisateurs.

---

### 5. Health Check Global

**GET** `/health`

Retourne l'état de santé global de l'API et de tous les GPUs.

**Paramètres :** Aucun

**Réponse :**
```json
{
  "status": "healthy",
  "service": "local-llm-api",
  "total_gpus": 2,
  "gpus": {
    "0": {
      "gpu_id": 0,
      "gpu_identifier": "GPU-0",
      "available": true,
      "model_loaded": true,
      "model_name": "gpt2",
      "device": "cuda:0",
      "metrics": {
        "memory": {
          "allocated_gb": 1.2,
          "reserved_gb": 1.5,
          "total_gb": 8.0,
          "free_gb": 6.5,
          "utilization_percent": 18.75
        },
        "gpu_info": {
          "name": "AMD Radeon RX Vega 64",
          "total_memory_gb": 8.0
        }
      }
    },
    "1": {
      "gpu_id": 1,
      "gpu_identifier": "GPU-1",
      "available": true,
      "model_loaded": false,
      "model_name": null,
      "device": "cuda:1",
      "metrics": {...}
    }
  },
  "summary": {
    "total_gpus_available": 2,
    "gpus_with_models": 1,
    "gpus_free": 1
  }
}
```

**Codes de statut :**
- `200` : Succès
- `500` : Erreur serveur

---

### 6. Health Check GPU Spécifique

**GET** `/health/<gpu_id>`

Retourne l'état de santé détaillé d'un GPU spécifique.

**Paramètres URL :**
- `gpu_id` (int) : Numéro du GPU (0 ou 1)

**Réponse :**
```json
{
  "status": "in_use",
  "gpu_id": 0,
  "gpu_identifier": "GPU-0",
  "available": true,
  "model": {
    "loaded": true,
    "name": "gpt2",
    "has_access_token": true
  },
  "device": "cuda:0",
  "message": "GPU 0 a un modèle chargé: gpt2",
  "metrics": {
    "memory": {
      "allocated_gb": 1.2,
      "reserved_gb": 1.5,
      "total_gb": 8.0,
      "free_gb": 6.5,
      "utilization_percent": 18.75
    },
    "gpu_info": {
      "name": "AMD Radeon RX Vega 64",
      "total_memory_gb": 8.0
    }
  }
}
```

**Statuts possibles :**
- `healthy` : API opérationnelle (vue globale uniquement)
- `free` : GPU disponible et libre
- `in_use` : GPU utilisé avec un modèle chargé
- `unavailable` : GPU non disponible

**Codes de statut :**
- `200` : Succès
- `400` : GPU ID invalide
- `500` : Erreur serveur

---

### 7. Chat Conversationnel

**POST** `/chat/<gpu_id>`

Génère une réponse de chat conversationnel en utilisant le modèle chargé sur le GPU spécifié.

**Paramètres URL :**
- `gpu_id` (int) : Numéro du GPU (0 ou 1) contenant le modèle à utiliser

**Body JSON requis :**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Tu es un assistant utile."
    },
    {
      "role": "user",
      "content": "Bonjour!"
    },
    {
      "role": "assistant",
      "content": "Bonjour! Comment puis-je vous aider?"
    },
    {
      "role": "user",
      "content": "Quelle est la capitale de la France?"
    }
  ],
  "temperature": 0.7,
  "max_new_tokens": 512
}
```

**Paramètres Body :**
- `messages` (list, **requis**) : Liste de messages conversationnels
  - Chaque message doit être un objet avec :
    - `role` (str) : `"system"`, `"user"` ou `"assistant"`
    - `content` (str) : Contenu du message
- `temperature` (float, optionnel) : Température pour la génération (0.0-2.0, défaut: 0.7)
- `max_new_tokens` (int, optionnel) : Nombre maximum de nouveaux tokens à générer (1-4096, défaut: 512)

**Réponse succès (200) :**
```json
{
  "status": "success",
  "response": "La capitale de la France est Paris.",
  "gpu_id": 0,
  "model_name": "gpt2",
  "parameters": {
    "temperature": 0.7,
    "max_new_tokens": 512
  }
}
```

**Réponse erreur (400) - Messages invalides :**
```json
{
  "status": "error",
  "message": "Le champ 'messages' doit être une liste de messages conversationnels [{'role': '...', 'content': '...'}]."
}
```

**Réponse erreur (404) - Modèle non chargé :**
```json
{
  "status": "error",
  "message": "Aucun modèle n'est chargé sur le GPU 0. Chargez un modèle avec POST /models/load/{model_name} d'abord."
}
```

**Codes de statut :**
- `200` : Réponse générée avec succès
- `400` : Erreur de validation (paramètres manquants ou invalides)
- `404` : GPU non disponible ou modèle non chargé
- `500` : Erreur lors de la génération

---

### 8. Completion de Texte

**POST** `/completion/<gpu_id>`

Génère une completion de texte simple en utilisant le modèle chargé sur le GPU spécifié.

**Paramètres URL :**
- `gpu_id` (int) : Numéro du GPU (0 ou 1) contenant le modèle à utiliser

**Body JSON requis :**
```json
{
  "prompt": "Le machine learning est une branche de",
  "temperature": 0.7,
  "max_new_tokens": 512
}
```

**Paramètres Body :**
- `prompt` (str, **requis**) : Prompt texte simple à compléter
- `temperature` (float, optionnel) : Température pour la génération (0.0-2.0, défaut: 0.7)
- `max_new_tokens` (int, optionnel) : Nombre maximum de nouveaux tokens à générer (1-4096, défaut: 512)

**Réponse succès (200) :**
```json
{
  "status": "success",
  "response": "l'intelligence artificielle qui permet aux machines d'apprendre...",
  "gpu_id": 0,
  "model_name": "gpt2",
  "parameters": {
    "temperature": 0.7,
    "max_new_tokens": 512
  }
}
```

**Réponse erreur (400) - Prompt invalide :**
```json
{
  "status": "error",
  "message": "Le champ 'prompt' doit être une chaîne de caractères (str)."
}
```

**Réponse erreur (404) - Modèle non chargé :**
```json
{
  "status": "error",
  "message": "Aucun modèle n'est chargé sur le GPU 0. Chargez un modèle avec POST /models/load/{model_name} d'abord."
}
```

**Codes de statut :**
- `200` : Réponse générée avec succès
- `400` : Erreur de validation (paramètres manquants ou invalides)
- `404` : GPU non disponible ou modèle non chargé
- `500` : Erreur lors de la génération

---

## Exemples d'utilisation avec cURL

### 1. Vérifier que l'API est active
```bash
curl http://localhost:5000/
```

### 2. Lister les modèles disponibles
```bash
curl http://localhost:5000/models
```

### 3. Charger un modèle
```bash
curl -X POST http://localhost:5000/models/load/gpt2 \
  -H "Content-Type: application/json"
```

### 4. Vérifier l'état de santé
```bash
curl http://localhost:5000/health
curl http://localhost:5000/health/0
```

### 5. Utiliser le chat
```bash
curl -X POST http://localhost:5000/chat/0 \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Bonjour, comment ça va?"}
    ],
    "temperature": 0.7,
    "max_new_tokens": 100
  }'
```

### 6. Utiliser la completion
```bash
curl -X POST http://localhost:5000/completion/0 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Le Python est un langage de programmation",
    "temperature": 0.7,
    "max_new_tokens": 50
  }'
```

### 7. Décharger un modèle
```bash
curl -X POST http://localhost:5000/models/unload/0 \
  -H "Content-Type: application/json" \
  -d '{
    "access_token": "votre_token_ici"
  }'
```

---

## Notes importantes

1. **GPU IDs** : Les GPU sont identifiés par 0 ou 1. Assurez-vous que le GPU spécifié existe et a un modèle chargé avant d'utiliser `/chat` ou `/completion`.

2. **Access Tokens** : Lors du chargement d'un modèle, un token d'accès unique est généré. **Conservez-le** car il est nécessaire pour décharger le modèle plus tard. Ce système protège vos modèles contre le déchargement par d'autres utilisateurs.

3. **Température** : 
   - Valeurs basses (0.0-0.3) : Réponses plus déterministes et cohérentes
   - Valeurs moyennes (0.7-0.9) : Bon équilibre créativité/cohérence
   - Valeurs hautes (1.0-2.0) : Réponses plus créatives mais moins prévisibles

4. **Max New Tokens** : Limite le nombre de tokens générés. Augmentez pour des réponses plus longues, mais cela augmente aussi le temps de génération.

5. **Format des messages (chat)** : Le format conversationnel suit le standard avec les rôles `system`, `user`, et `assistant`. L'historique complet est envoyé à chaque requête.

---

## Codes d'erreur HTTP

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 400 | Requête invalide (paramètres manquants ou incorrects) |
| 403 | Accès refusé (token invalide) |
| 404 | Ressource introuvable (modèle, GPU, etc.) |
| 500 | Erreur serveur interne |
| 503 | Service indisponible (aucun GPU libre) |

---

## Support

Pour plus d'informations, consultez le code source ou les logs de l'API pour les détails d'erreur.

