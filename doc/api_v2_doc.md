# Documentation API Overlay

Cette documentation décrit tous les endpoints disponibles de l'API Overlay pour la gestion des conversations, messages et providers LLM.

## Informations générales

- **Base URL** : `http://localhost:8000` (par défaut)
- **Format des réponses** : JSON
- **Rate Limiting** :
  - Global : 200 requêtes/jour, 50 requêtes/heure
  - Création de conversation : 10 requêtes/minute
  - Envoi de message : 30 requêtes/minute
- **CORS** : Activé (configurable via variable d'environnement `CORS_ORIGINS`)

---

## Table des matières

1. [Status](#status)
2. [Providers](#providers)
3. [Conversations](#conversations)
4. [Messages](#messages)

---

## Status

### GET `/`

Récupère le statut de l'API et de la connexion à la Base API.

**Réponse 200** :
```json
{
  "base-api": {
    "message": "Base API connected",
    "status": 200
  },
  "overlay-api": {
    "message": "Overlay API ON",
    "status": 200
  }
}
```

**Réponse 503** (si Base API non connectée) :
```json
{
  "base-api": {
    "message": "Base API not connected",
    "status": 503
  },
  "overlay-api": {
    "message": "Overlay API ON",
    "status": 200
  }
}
```

---

## Providers

### GET `/providers`

Liste tous les providers disponibles avec leur statut de disponibilité.

**Réponse 200** :
```json
{
  "providers": [
    {
      "name": "local",
      "available": true
    },
    {
      "name": "groq",
      "available": true
    }
  ],
  "count": 2
}
```

**Réponse 500** :
```json
{
  "error": "Internal server error"
}
```

---

### GET `/providers/<provider_name>/models`

Récupère la liste des modèles disponibles pour un provider spécifique.

**Paramètres de chemin** :
- `provider_name` (string, requis) : Nom du provider (`local` ou `groq`)

**Réponse 200** :
```json
{
  "provider": "local",
  "models": [
    {
      "identifier": "mistral"
    }
  ],
  "count": 1
}
```

**Réponse 400** (Provider invalide) :
```json
{
  "error": "Provider 'invalid' non reconnu. Utilisez 'local' ou 'groq'"
}
```

**Réponse 503** (Provider non disponible) :
```json
{
  "error": "Provider 'local' n'est pas disponible",
  "provider": "local"
}
```

**Réponse 500** :
```json
{
  "error": "Erreur lors de la récupération des modèles",
  "details": "...",
  "provider": "local"
}
```

---

## Conversations

### GET `/conversations`

Récupère la liste de toutes les conversations.

**Réponse 200** :
```json
[
  {
    "id": 1,
    "name": "Ma conversation",
    "model_name": "mistral",
    "provider": "local",
    "temperature": 0.7,
    "message_max": 10,
    "created_at": "2024-01-01T12:00:00",
    "updated_at": "2024-01-01T12:00:00"
  }
]
```

**Réponse 500** :
```json
{
  "error": "..."
}
```

---

### GET `/conversations/<id>`

Récupère une conversation spécifique par son ID.

**Paramètres de chemin** :
- `id` (integer, requis) : ID de la conversation

**Réponse 200** :
```json
{
  "id": 1,
  "name": "Ma conversation",
  "model_name": "mistral",
  "provider": "local",
  "temperature": 0.7,
  "message_max": 10,
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:00"
}
```

**Réponse 404** :
```json
{
  "error": "Conversation 1 not found"
}
```

**Réponse 500** :
```json
{
  "error": "..."
}
```

---

### POST `/conversations`

Crée une nouvelle conversation.

**Rate Limit** : 10 requêtes/minute

**Corps de la requête** (JSON) :
```json
{
  "model_name": "mistral",
  "provider": "local",
  "name": "Ma conversation",
  "temperature": 0.7,
  "message_max": 10,
  "system_prompt": "Tu es un assistant utile. Réponds toujours en français."
}
```

**Champs** :
- `model_name` (string, **requis**) : Nom du modèle à utiliser (1-200 caractères)
- `provider` (string, optionnel) : Provider à utiliser (`local` ou `groq`, défaut: `local`)
- `name` (string, optionnel) : Nom de la conversation (1-200 caractères, défaut: `"Conversation"`)
- `temperature` (float, optionnel) : Température pour la génération (0.0-2.0, défaut: `0.7`)
- `message_max` (integer, optionnel) : Nombre maximum de messages (0-1000, défaut: `10`)
- `system_prompt` (string, optionnel) : Message système pour guider le modèle (max 2000 caractères)

**Réponse 201** :
```json
{
  "id": 1,
  "name": "Ma conversation",
  "model_name": "mistral",
  "provider": "local",
  "temperature": 0.7,
  "message_max": 10,
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:00"
}
```

**Réponse 400** (Erreur de validation) :
```json
{
  "error": "Validation error",
  "details": [
    "model_name: ne peut pas être vide",
    "temperature: ensure this value is less than or equal to 2.0"
  ]
}
```

**Réponse 400** (Body manquant) :
```json
{
  "error": "Request body is required"
}
```

**Réponse 500** :
```json
{
  "error": "Internal server error"
}
```

---

### DELETE `/conversations`

Supprime toutes les conversations.

**Réponse 200** :
```json
{
  "message": "5 conversation(s) deleted",
  "count": 5
}
```

**Réponse 500** :
```json
{
  "error": "..."
}
```

---

### DELETE `/conversations/<id>`

Supprime une conversation spécifique par son ID.

**Paramètres de chemin** :
- `id` (integer, requis) : ID de la conversation à supprimer

**Réponse 200** :
```json
{
  "message": "Conversation 1 deleted"
}
```

**Réponse 404** :
```json
{
  "error": "Conversation 1 not found"
}
```

**Réponse 500** :
```json
{
  "error": "..."
}
```

---

## Messages

### GET `/conversations/<id>/message`

Récupère tous les messages d'une conversation.

**Paramètres de chemin** :
- `id` (integer, requis) : ID de la conversation

**Réponse 200** :
```json
[
  {
    "id": 1,
    "conversation_id": 1,
    "role": "user",
    "content": "Bonjour, comment ça va ?",
    "created_at": "2024-01-01T12:00:00"
  },
  {
    "id": 2,
    "conversation_id": 1,
    "role": "assistant",
    "content": "Bonjour ! Je vais bien, merci.",
    "created_at": "2024-01-01T12:00:05"
  }
]
```

**Réponse 500** :
```json
{
  "error": "..."
}
```

---

### GET `/conversations/<id>/message/<message_id>`

Récupère un message spécifique d'une conversation.

**Paramètres de chemin** :
- `id` (integer, requis) : ID de la conversation
- `message_id` (integer, requis) : ID du message

**Réponse 200** :
```json
{
  "id": 1,
  "conversation_id": 1,
  "role": "user",
  "content": "Bonjour, comment ça va ?",
  "created_at": "2024-01-01T12:00:00"
}
```

**Réponse 404** :
```json
{
  "error": "Message not found"
}
```

**Réponse 500** :
```json
{
  "error": "..."
}
```

---

### POST `/conversations/<id>/message`

Envoie un message dans une conversation. Le message sera traité par le modèle LLM configuré pour la conversation et une réponse sera générée.

**Rate Limit** : 30 requêtes/minute

**Paramètres de chemin** :
- `id` (integer, requis) : ID de la conversation

**Paramètres de requête (query)** :
- `provider` (string, optionnel) : Surcharge du provider pour ce message uniquement (`local` ou `groq`)

**Corps de la requête** (JSON) :
```json
{
  "content": "Bonjour, comment ça va ?"
}
```

**Champs** :
- `content` (string, **requis**) : Contenu du message (1-100000 caractères, les espaces en début/fin sont automatiquement supprimés)

**Exemple avec surcharge de provider** :
```
POST /conversations/1/message?provider=groq
```

**Réponse 200** :
```json
{
  "id": 2,
  "conversation_id": 1,
  "role": "assistant",
  "content": "Bonjour ! Je vais bien, merci. Comment puis-je vous aider ?",
  "created_at": "2024-01-01T12:00:05"
}
```

**Réponse 400** (Erreur de validation) :
```json
{
  "error": "Validation error",
  "details": [
    "content: ne peut pas être vide"
  ]
}
```

**Réponse 400** (Provider invalide dans query) :
```json
{
  "error": "provider must be 'local' or 'groq'"
}
```

**Réponse 400** (Body manquant) :
```json
{
  "error": "Request body is required"
}
```

**Réponse 404** (Conversation non trouvée) :
```json
{
  "error": "Conversation 1 not found"
}
```

**Réponse 500** :
```json
{
  "error": "Internal server error"
}
```

---

### DELETE `/conversations/<id>/message/<message_id>`

Supprime un message spécifique d'une conversation.

**Paramètres de chemin** :
- `id` (integer, requis) : ID de la conversation
- `message_id` (integer, requis) : ID du message à supprimer

**Réponse 200** :
```json
{
  "message": "Message 1 deleted"
}
```

**Réponse 404** :
```json
{
  "error": "Message 1 not found"
}
```

**Réponse 500** :
```json
{
  "error": "..."
}
```

---

## Codes de statut HTTP

| Code | Description |
|------|-------------|
| 200 | Succès |
| 201 | Ressource créée avec succès |
| 400 | Erreur de validation ou requête invalide |
| 404 | Ressource non trouvée |
| 500 | Erreur serveur interne |
| 503 | Service non disponible (provider non disponible, Base API non connectée) |

---

## Exemples d'utilisation

### Créer une conversation et envoyer un message

```bash
# 1. Créer une conversation
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mistral",
    "provider": "local",
    "name": "Ma première conversation",
    "temperature": 0.7
  }'

# Réponse: {"id": 1, ...}

# 2. Envoyer un message
curl -X POST http://localhost:8000/conversations/1/message \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Bonjour, peux-tu me parler de Python ?"
  }'

# Réponse: {"id": 2, "role": "assistant", "content": "...", ...}

# 3. Récupérer tous les messages
curl http://localhost:8000/conversations/1/message
```

### Lister les providers et leurs modèles

```bash
# 1. Lister les providers
curl http://localhost:8000/providers

# 2. Lister les modèles d'un provider
curl http://localhost:8000/providers/local/models
```

---

## Notes importantes

1. **Rate Limiting** : Les limites de taux sont appliquées par adresse IP. Si vous dépassez les limites, vous recevrez une réponse 429 (Too Many Requests).

2. **Validation** : Tous les champs sont validés côté serveur. Les erreurs de validation retournent un code 400 avec les détails des erreurs.

3. **Provider** : Les providers supportés sont `local` et `groq`. Le provider par défaut est `local` si non spécifié.

4. **Messages système** : Si aucun `system_prompt` n'est fourni lors de la création d'une conversation, un prompt par défaut sera utilisé.

5. **Surcharge de provider** : Il est possible de surcharger le provider pour un message spécifique en utilisant le paramètre de requête `?provider=groq` sur l'endpoint POST `/conversations/<id>/message`.

6. **CORS** : L'API supporte les requêtes cross-origin. Configurez `CORS_ORIGINS` pour restreindre les origines autorisées en production.

