# Rapport sur l'État du Backend - Local LLM API

**Date :** 2025-01-27  
**Projet :** Local LLM API Backend  
**Plateforme cible :** Ubuntu Linux avec 2 GPUs AMD Vega 64 (LBB)

---

## 1. État Actuel du Backend

### 1.1 Ce qui existe ✅

#### A. Client Python (`src/client/`)
- ✅ **Client complet** : Classe `LLMClient` avec tous les endpoints
- ✅ **Endpoints implémentés** :
  - `root.py` - Route racine
  - `models.py` - Liste, chargement, déchargement de modèles
  - `health.py` - Health checks global et par GPU
  - `chat.py` - Génération de chat
  - `completion.py` - Completion de texte
  - `logs.py` - Streaming, historique et statistiques des logs
- ✅ **Gestion des erreurs** : Exceptions personnalisées
- ✅ **Configuration** : Support des variables d'environnement et configuration personnalisée

#### B. Système de Conversation (`src/conversation/`)
- ✅ **Gestionnaire de conversations** : `ConversationManager`
- ✅ **Gestion du contexte** : `ConversationContext` avec troncature
- ✅ **Templates** : Support pour TinyLlama, FinBERT, Qwen2.5
- ✅ **Persistance SQLite** : Stockage des conversations
- ✅ **Registre de templates** : Détection automatique par nom de modèle

#### C. Documentation
- ✅ **Documentation API complète** : `doc/API_DOCUMENTATION.md`
- ✅ **Architecture surcouche** : `doc/ARCHITECTURE_SURCOUCHE.md`
- ✅ **Guide d'implémentation** : `doc/IMPLÉMENTATION_SURCOUCHE.md`
- ✅ **Exemples d'utilisation** : `doc/exemple_utilisation_conversation.py`

#### D. Tests
- ✅ **Tests client** : `src/tests/test_client.py` (modifié pour GPU1)

### 1.2 Ce qui manque ❌

#### A. Backend Flask (`src/main.py`)
**PROBLÈME CRITIQUE** : Le backend Flask est **incomplet**.

**État actuel :**
- ✅ Route racine (`GET /`) - **Implémentée**
- ❌ Toutes les autres routes - **MANQUANTES**

**Routes manquantes :**
1. `GET /models` - Liste des modèles disponibles
2. `POST /models/load/<model_name>` - Charger un modèle
3. `POST /models/unload/<gpu_id>` - Décharger un modèle
4. `GET /health` - Health check global
5. `GET /health/<gpu_id>` - Health check par GPU
6. `POST /chat/<gpu_id>` - Génération de chat
7. `POST /completion/<gpu_id>` - Completion de texte
8. `GET /logs/stream` - Streaming des logs (SSE)
9. `GET /logs/history` - Historique des logs
10. `GET /logs/stats` - Statistiques des logs

**Fonctionnalités backend manquantes :**
- ❌ Gestionnaire LLM (chargement/déchargement de modèles sur GPU)
- ❌ Gestion des GPUs (détection, allocation, monitoring)
- ❌ Système de logging avec buffer circulaire
- ❌ Gestion des access tokens pour la sécurité
- ❌ Intégration avec PyTorch/ROCm pour AMD GPUs
- ❌ Gestion des modèles Hugging Face (cache, téléchargement)

---

## 2. Analyse des Scripts Manquants

### 2.1 Scripts Backend Nécessaires

#### A. Gestionnaire LLM (`src/backend/llm_manager.py`)
**Fonctionnalités requises :**
- Charger un modèle Hugging Face sur un GPU spécifique
- Décharger un modèle d'un GPU
- Gérer les modèles chargés (état, métadonnées)
- Gérer les access tokens pour la sécurité
- Détecter les modèles disponibles dans le cache Hugging Face
- Support PyTorch 2.2.0 avec ROCm 5.7

**Dépendances :**
- `torch` (2.2.0)
- `transformers` (Hugging Face)
- `huggingface_hub`

#### B. Gestionnaire GPU (`src/backend/gpu_manager.py`)
**Fonctionnalités requises :**
- Détecter les GPUs disponibles (AMD via ROCm)
- Vérifier l'état de chaque GPU (mémoire, utilisation)
- Allouer un GPU libre pour un modèle
- Retourner les métriques GPU (mémoire allouée, réservée, totale)
- Gérer les identifiants GPU (GPU-0, GPU-1)

**Dépendances :**
- `torch` avec support ROCm
- `pynvml` ou équivalent AMD (si disponible)

#### C. Système de Logging (`src/backend/logging_handler.py`)
**Fonctionnalités requises :**
- Buffer circulaire en mémoire (max 1000 logs)
- Handler personnalisé pour capturer les logs de l'application
- Support Server-Sent Events (SSE) pour le streaming
- Filtrage par niveau (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Statistiques des logs

**Dépendances :**
- `logging` (stdlib)
- `flask` avec support SSE

#### D. Routes Flask (`src/backend/routes/`)
**Structure recommandée :**
```
src/backend/routes/
├── __init__.py
├── models.py      # Routes /models, /models/load, /models/unload
├── health.py      # Routes /health, /health/<gpu_id>
├── chat.py        # Route /chat/<gpu_id>
├── completion.py  # Route /completion/<gpu_id>
└── logs.py        # Routes /logs/stream, /logs/history, /logs/stats
```

#### E. Configuration (`src/backend/config.py`)
**Fonctionnalités requises :**
- Configuration par défaut
- Variables d'environnement
- Paramètres ROCm/PyTorch
- Chemins de cache Hugging Face
- Port et host du serveur

#### F. Application Flask Principale (`src/main.py` - à compléter)
**Fonctionnalités requises :**
- Initialisation de l'application Flask
- Enregistrement des blueprints/routes
- Configuration CORS (si nécessaire)
- Gestion des erreurs globales
- Initialisation des managers (LLM, GPU, Logging)
- Point d'entrée pour `python -m src.main`

---

## 3. Tests Nécessaires

### 3.1 Tests Backend Manquants

#### A. Tests Unitaires
- ❌ `tests/backend/test_llm_manager.py` - Tests du gestionnaire LLM
- ❌ `tests/backend/test_gpu_manager.py` - Tests du gestionnaire GPU
- ❌ `tests/backend/test_logging_handler.py` - Tests du système de logging

#### B. Tests d'Intégration
- ❌ `tests/integration/test_api_routes.py` - Tests de toutes les routes API
- ❌ `tests/integration/test_model_lifecycle.py` - Tests du cycle complet (load → use → unload)
- ❌ `tests/integration/test_multi_gpu.py` - Tests avec plusieurs GPUs

#### C. Tests de Performance
- ❌ `tests/performance/test_model_loading.py` - Temps de chargement
- ❌ `tests/performance/test_concurrent_requests.py` - Requêtes simultanées

### 3.2 Tests Client Existants
- ✅ `src/tests/test_client.py` - Tests du client Python (modifié pour GPU1)

**Recommandation :** Les tests client existants ne peuvent pas être exécutés car le backend n'est pas implémenté. Une fois le backend complet, ces tests pourront valider l'intégration.

---

## 4. Routes API à Implémenter

### 4.1 Routes Prioritaires (Minimum Viable)

#### Phase 1 : Fonctionnalités de base
1. ✅ `GET /` - Route racine (déjà implémentée)
2. 🔴 `GET /models` - Liste des modèles disponibles
3. 🔴 `POST /models/load/<model_name>` - Charger un modèle
4. 🔴 `POST /models/unload/<gpu_id>` - Décharger un modèle
5. 🔴 `GET /health` - Health check global
6. 🔴 `GET /health/<gpu_id>` - Health check par GPU
7. 🔴 `POST /chat/<gpu_id>` - Génération de chat
8. 🔴 `POST /completion/<gpu_id>` - Completion de texte

#### Phase 2 : Monitoring et logs
9. 🔴 `GET /logs/stream` - Streaming des logs (SSE)
10. 🔴 `GET /logs/history` - Historique des logs
11. 🔴 `GET /logs/stats` - Statistiques des logs

### 4.2 Routes Supplémentaires Recommandées (Pour usage production)

#### A. Routes de Conversation (utilisant la surcouche)
- `POST /conversations` - Créer une nouvelle conversation
- `POST /conversations/<conversation_id>/messages` - Envoyer un message dans une conversation
- `GET /conversations/<conversation_id>` - Récupérer une conversation
- `GET /conversations` - Lister les conversations
- `DELETE /conversations/<conversation_id>` - Supprimer une conversation
- `POST /conversations/<conversation_id>/clear` - Vider le contexte d'une conversation

**Avantages :**
- Utilise le système de conversation existant (`ConversationManager`)
- Gestion automatique du contexte et des templates
- Persistance SQLite intégrée
- Support multi-modèles avec templates adaptés

#### B. Routes de Gestion des Modèles
- `GET /models/<model_name>/info` - Informations détaillées sur un modèle
- `POST /models/<model_name>/download` - Télécharger un modèle depuis Hugging Face
- `GET /models/cache` - Statistiques du cache Hugging Face

#### C. Routes de Monitoring Avancé
- `GET /metrics` - Métriques Prometheus (pour monitoring)
- `GET /gpus/<gpu_id>/metrics` - Métriques détaillées d'un GPU
- `GET /stats` - Statistiques globales de l'API (requêtes, temps de réponse, etc.)

#### D. Routes de Configuration
- `GET /config` - Configuration actuelle (sans secrets)
- `PUT /config` - Modifier la configuration (restart nécessaire)

#### E. Routes de Streaming (Amélioration)
- `POST /chat/<gpu_id>/stream` - Chat avec streaming de la réponse (token par token)
- `POST /completion/<gpu_id>/stream` - Completion avec streaming

**Avantages du streaming :**
- Meilleure UX pour les applications web
- Réponses plus rapides perçues par l'utilisateur
- Support des réponses longues sans timeout

---

## 5. Architecture Backend Recommandée

### 5.1 Structure de Fichiers

```
src/
├── main.py                    # Application Flask principale (à compléter)
├── backend/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── app.py                 # Factory Flask (optionnel)
│   ├── llm_manager.py         # Gestionnaire LLM
│   ├── gpu_manager.py         # Gestionnaire GPU
│   ├── logging_handler.py     # Système de logging
│   ├── auth.py                # Gestion des access tokens
│   └── routes/
│       ├── __init__.py
│       ├── models.py
│       ├── health.py
│       ├── chat.py
│       ├── completion.py
│       └── logs.py
├── client/                    # Client Python (existant)
│   └── ...
├── conversation/              # Système de conversation (existant)
│   └── ...
└── tests/
    ├── backend/
    │   ├── test_llm_manager.py
    │   ├── test_gpu_manager.py
    │   └── test_logging_handler.py
    ├── integration/
    │   ├── test_api_routes.py
    │   └── test_model_lifecycle.py
    └── test_client.py         # Tests client (existant)
```

### 5.2 Dépendances Backend

**Fichier `requirements.txt` (à compléter) :**
```txt
flask>=3.0.0
flask-cors>=4.0.0
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
transformers>=4.30.0
huggingface-hub>=0.16.0
numpy==1.26.4
```

**Note :** ROCm 5.7 doit être installé au niveau système (pas via pip).

---

## 6. Plan d'Implémentation Recommandé

### Phase 1 : Backend Core (Priorité Maximale)
1. **Gestionnaire GPU** (`gpu_manager.py`)
   - Détection des GPUs AMD
   - Métriques GPU (mémoire, utilisation)
   - Allocations GPU

2. **Gestionnaire LLM** (`llm_manager.py`)
   - Chargement de modèles Hugging Face
   - Gestion des modèles en mémoire
   - Access tokens

3. **Routes de base** (`routes/models.py`, `routes/health.py`)
   - `GET /models`
   - `POST /models/load/<model_name>`
   - `POST /models/unload/<gpu_id>`
   - `GET /health`
   - `GET /health/<gpu_id>`

4. **Routes de génération** (`routes/chat.py`, `routes/completion.py`)
   - `POST /chat/<gpu_id>`
   - `POST /completion/<gpu_id>`

### Phase 2 : Monitoring et Logs
5. **Système de logging** (`logging_handler.py`)
   - Buffer circulaire
   - Handler personnalisé

6. **Routes de logs** (`routes/logs.py`)
   - `GET /logs/stream`
   - `GET /logs/history`
   - `GET /logs/stats`

### Phase 3 : Tests
7. **Tests unitaires** - Tous les managers
8. **Tests d'intégration** - Routes API
9. **Tests avec le client** - Validation end-to-end

### Phase 4 : Routes Supplémentaires (Optionnel)
10. **Routes de conversation** - Utilisation de `ConversationManager`
11. **Streaming** - Chat et completion avec streaming
12. **Monitoring avancé** - Métriques Prometheus

---

## 7. Points d'Attention Techniques

### 7.1 Compatibilité PyTorch/ROCm
- **PyTorch 2.2.0** avec support ROCm 5.7
- Vérifier que les modèles Hugging Face sont compatibles
- Gérer les différences entre CUDA et ROCm (device: `cuda:0` vs `cuda:0` sur ROCm)

### 7.2 Gestion Mémoire GPU
- **AMD Vega 64** : 8 GB VRAM par GPU
- Surveiller l'utilisation mémoire pour éviter les OOM
- Gérer le déchargement propre des modèles
- MIOpen cache actif (4 Go) - prendre en compte dans les calculs

### 7.3 Access Tokens
- Générer des tokens uniques et sécurisés
- Stocker les tokens associés aux modèles chargés
- Valider les tokens lors du déchargement
- Expiration des tokens (optionnel, pour sécurité)

### 7.4 Concurrence
- Gérer plusieurs requêtes simultanées
- Lock sur les opérations GPU (chargement/déchargement)
- Queue pour les requêtes de génération (si nécessaire)

### 7.5 Timeouts
- Timeout long pour le chargement de modèles (10+ minutes)
- Timeout pour la génération (variable selon max_new_tokens)
- Gestion des timeouts côté client et serveur

---

## 8. Résumé des Actions Requises

### Actions Immédiates (Critiques)
1. ✅ **Modifier `test_client.py` pour GPU1** - TERMINÉ
2. 🔴 **Implémenter `gpu_manager.py`** - MANQUANT
3. 🔴 **Implémenter `llm_manager.py`** - MANQUANT
4. 🔴 **Implémenter toutes les routes API** - MANQUANT
5. 🔴 **Compléter `src/main.py`** - MANQUANT

### Actions Recommandées (Important)
6. 🔴 **Implémenter le système de logging** - MANQUANT
7. 🔴 **Créer les tests unitaires** - MANQUANT
8. 🔴 **Créer les tests d'intégration** - MANQUANT
9. 🔴 **Documenter les endpoints backend** - MANQUANT

### Actions Optionnelles (Amélioration)
10. 🔴 **Implémenter les routes de conversation** - MANQUANT
11. 🔴 **Ajouter le streaming** - MANQUANT
12. 🔴 **Ajouter le monitoring avancé** - MANQUANT

---

## 9. Conclusion

### État Actuel
Le projet a une **base solide** avec :
- ✅ Client Python complet et fonctionnel
- ✅ Système de conversation avec templates
- ✅ Documentation complète
- ❌ **Backend Flask incomplet** (seulement route racine)

### Blocage Principal
**Le backend Flask n'est pas implémenté.** Toutes les routes API décrites dans la documentation n'existent pas dans le code. Le client Python ne peut pas fonctionner car il n'y a pas de serveur pour répondre aux requêtes.

### Prochaines Étapes
1. **Implémenter le backend Flask complet** (priorité absolue)
2. **Tester avec le client Python existant**
3. **Ajouter les routes de conversation** (pour utiliser la surcouche)
4. **Améliorer avec streaming et monitoring**

### Estimation
- **Backend Core** : ~2-3 semaines (selon complexité PyTorch/ROCm)
- **Tests** : ~1 semaine
- **Routes supplémentaires** : ~1 semaine
- **Total** : ~4-5 semaines pour un backend complet et testé

---

**Rapport généré le :** 2025-01-27  
**Auteur :** Assistant IA  
**Version :** 1.0

