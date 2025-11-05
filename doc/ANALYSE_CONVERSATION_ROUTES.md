# Analyse du module Conversation - Routes API recommandées

## 📋 Fonctionnalités identifiées

### 1. Gestion des conversations
- **Création** : Créer une conversation avec modèle, GPU, system prompt, limite de messages
- **Récupération** : Obtenir une conversation par ID (avec contexte complet)
- **Liste** : Lister les conversations (filtrées par modèle, limitées)
- **Suppression** : Supprimer une conversation complète
- **Mise à jour** : Modifier les paramètres (max_messages)

### 2. Envoi de messages
- **Envoi avec réponse** : Envoyer un message utilisateur et recevoir une réponse assistant
- **Formatage automatique** : Formatage selon le template du modèle (TinyLlama, Qwen, Mistral, etc.)
- **Gestion du contexte** : Historique automatique avec troncature intelligente
- **Paramètres de génération** : Temperature, max_new_tokens

### 3. Gestion du contexte
- **Vider le contexte** : Supprimer l'historique (garder ou supprimer le système)
- **Troncature** : Limite automatique de messages (max_messages)
- **Résumé** : Support prévu pour résumé des anciens messages (futur)

### 4. Persistance
- **Auto-save** : Sauvegarde automatique après chaque message
- **SQLite** : Stockage local avec base de données SQLite
- **Métadonnées** : Timestamps, modèles, GPU, etc.

---

## 🛣️ Routes API recommandées

### Groupe 1 : Gestion des conversations (CRUD)

#### 1.1 Créer une conversation
**POST** `/conversations`
- **Body JSON** :
  ```json
  {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "gpu_id": 0,
    "system_prompt": "Tu es un assistant utile.",  // optionnel
    "max_messages": 50,  // optionnel
    "conversation_id": "uuid-custom"  // optionnel (sinon généré)
  }
  ```
- **Réponse** :
  ```json
  {
    "status": "success",
    "conversation_id": "uuid-...",
    "model_name": "...",
    "gpu_id": 0,
    "created_at": "2024-01-15T10:00:00"
  }
  ```

#### 1.2 Récupérer une conversation
**GET** `/conversations/<conversation_id>`
- **Réponse** :
  ```json
  {
    "status": "success",
    "conversation": {
      "conversation_id": "...",
      "model_name": "...",
      "gpu_id": 0,
      "system_prompt": "...",
      "max_messages": 50,
      "message_count": 5,
      "created_at": "...",
      "updated_at": "...",
      "messages": [
        {
          "role": "user",
          "content": "...",
          "timestamp": "..."
        }
      ]
    }
  }
  ```

#### 1.3 Lister les conversations
**GET** `/conversations`
- **Query params** :
  - `model_name` (optionnel) : Filtrer par modèle
  - `limit` (optionnel) : Limiter le nombre de résultats
- **Réponse** :
  ```json
  {
    "status": "success",
    "count": 10,
    "conversations": [
      {
        "conversation_id": "...",
        "model_name": "...",
        "gpu_id": 0,
        "message_count": 5,
        "updated_at": "..."
      }
    ]
  }
  ```

#### 1.4 Supprimer une conversation
**DELETE** `/conversations/<conversation_id>`
- **Réponse** :
  ```json
  {
    "status": "success",
    "message": "Conversation supprimée avec succès"
  }
  ```

---

### Groupe 2 : Envoi de messages

#### 2.1 Envoyer un message (avec réponse)
**POST** `/conversations/<conversation_id>/messages`
- **Body JSON** :
  ```json
  {
    "message": "Bonjour, comment ça va ?",
    "temperature": 0.7,  // optionnel
    "max_new_tokens": 150  // optionnel
  }
  ```
- **Réponse** :
  ```json
  {
    "status": "success",
    "response": "Bonjour ! Je vais bien, merci.",
    "conversation_id": "...",
    "message_count": 6,
    "parameters": {
      "temperature": 0.7,
      "max_new_tokens": 150
    }
  }
  ```

---

### Groupe 3 : Gestion du contexte

#### 3.1 Vider le contexte
**DELETE** `/conversations/<conversation_id>/context`
- **Query params** :
  - `keep_system` (bool, défaut: true) : Garder les messages système
- **Réponse** :
  ```json
  {
    "status": "success",
    "message": "Contexte vidé avec succès",
    "kept_system": true
  }
  ```

#### 3.2 Modifier la limite de messages
**PATCH** `/conversations/<conversation_id>/max-messages`
- **Body JSON** :
  ```json
  {
    "max_messages": 100  // ou null pour pas de limite
  }
  ```
- **Réponse** :
  ```json
  {
    "status": "success",
    "max_messages": 100,
    "message": "Limite de messages mise à jour"
  }
  ```

---

## 📊 Routes prioritaires (MVP)

### Phase 1 - Essentiel (à implémenter en premier)
1. ✅ **POST** `/conversations` - Créer une conversation
2. ✅ **POST** `/conversations/<id>/messages` - Envoyer un message
3. ✅ **GET** `/conversations/<id>` - Récupérer une conversation
4. ✅ **GET** `/conversations` - Lister les conversations

### Phase 2 - Utile (à implémenter ensuite)
5. ✅ **DELETE** `/conversations/<id>` - Supprimer une conversation
6. ✅ **DELETE** `/conversations/<id>/context` - Vider le contexte
7. ✅ **PATCH** `/conversations/<id>/max-messages` - Modifier la limite

---

## 🔍 Détails techniques

### Gestion des erreurs
- **404** : Conversation introuvable
- **400** : Paramètres invalides (modèle non supporté, GPU invalide, etc.)
- **500** : Erreur serveur (API de base indisponible, erreur de stockage, etc.)

### Intégration avec l'API de base
- Le `ConversationManager` utilise `LLMClient` pour appeler l'API de base
- Le formatage des messages est géré automatiquement par les templates
- La persistance est gérée par `SQLiteStorage` (auto-save activé)

### Templates supportés
- TinyLlama (ChatML)
- Qwen2.5 (ChatML)
- Mistral 7B Instruct (format Mistral)
- Détection automatique via `TemplateRegistry`

---

## 💡 Recommandations

1. **Initialisation du ConversationManager** : 
   - Créer une instance globale dans `main.py` ou dans un module dédié
   - Utiliser `LLMClient` configuré pour appeler l'API de base (port 5000)

2. **Gestion des erreurs** :
   - Intercepter `ConversationNotFoundError` → 404
   - Intercepter `TemplateNotFoundError` → 400
   - Intercepter les erreurs de l'API de base → 502 ou 503

3. **Performance** :
   - Le cache en mémoire (`_active_conversations`) améliore les performances
   - SQLite est déjà optimisé avec WAL et index

4. **Sécurité** :
   - Valider les IDs de conversation (UUID)
   - Valider les paramètres (temperature, max_new_tokens)
   - Limiter la taille des messages (à implémenter)

