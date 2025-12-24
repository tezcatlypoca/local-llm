# 📚 API Local LLM - Documentation Complète

**Version:** 3.0.0  
**Base URL:** `http://localhost:5001`  
**Documentation interactive Swagger:** `http://localhost:5001/api-docs`

---

## 📑 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Authentification & Rate Limiting](#authentification--rate-limiting)
3. [Endpoints Généraux](#endpoints-généraux)
4. [Conversations](#conversations)
5. [Messages](#messages)
6. [Providers](#providers)
7. [RAG - Retrieval-Augmented Generation](#rag---retrieval-augmented-generation)
8. [Codes d'erreur](#codes-derreur)
9. [Exemples d'utilisation](#exemples-dutilisation)

---

## 🎯 Vue d'ensemble

Cette API fournit une interface complète pour :
- **Gérer des conversations** avec des modèles LLM (locaux ou via API)
- **Envoyer et recevoir des messages** avec support multi-provider
- **Interroger différents providers** (Local, Groq)
- **Système RAG** pour enrichir les réponses avec des connaissances externes

### Providers supportés
- **local** : Modèles LLM locaux (ex: Mistral, Qwen)
- **groq** : API Groq Cloud

---

## 🔐 Authentification & Rate Limiting

### CORS
- **Origins autorisés** : Configurable via `CORS_ORIGINS` (défaut: `*`)
- **Méthodes** : GET, POST, DELETE, OPTIONS
- **Headers** : Content-Type, Authorization

### Rate Limiting
- **Global** : 200 requêtes/jour, 50 requêtes/heure
- **Création conversation** : 10 requêtes/minute
- **Envoi message** : 30 requêtes/minute

---

## 🌐 Endpoints Généraux

### 🏠 Health Check

#### `GET /`
Vérifie l'état de l'API et la connexion à la base API.

**Réponse 200:**
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

---

## 💬 Conversations

### Créer une conversation

#### `POST /conversations`
Crée une nouvelle conversation avec un modèle LLM.

**Rate Limit:** 10 requêtes/minute

**Body (JSON):**
```json
{
  "model_name": "mistral",           // REQUIS
  "provider": "local",                // Optionnel (défaut: "local")
  "name": "Ma conversation",          // Optionnel
  "temperature": 0.7,                 // Optionnel (0.0-2.0, défaut: 0.7)
  "message_max": 10,                  // Optionnel (0-1000, défaut: 10)
  "system_prompt": "Tu es..."         // Optionnel (max 2000 caractères)
}
```

**Réponse 201:**
```json
{
  "id": 1,
  "model_name": "mistral",
  "provider": "local",
  "name": "Ma conversation",
  "temperature": 0.7,
  "message_max": 10,
  "system_prompt": "Tu es...",
  "created_at": "2025-12-24T10:30:00Z"
}
```

**Erreurs possibles:**
- `400` : Erreur de validation (champs manquants ou invalides)
- `500` : Erreur serveur

---

### Lister toutes les conversations

#### `GET /conversations`
Récupère la liste de toutes les conversations.

**Réponse 200:**
```json
[
  {
    "id": 1,
    "model_name": "mistral",
    "provider": "local",
    "name": "Ma conversation",
    "created_at": "2025-12-24T10:30:00Z"
  },
  {
    "id": 2,
    "model_name": "llama2",
    "provider": "groq",
    "name": "Test Groq",
    "created_at": "2025-12-24T11:00:00Z"
  }
]
```

---

### Récupérer une conversation

#### `GET /conversations/{id}`
Récupère les détails d'une conversation spécifique.

**Paramètres:**
- `id` (path, integer) : ID de la conversation

**Réponse 200:**
```json
{
  "id": 1,
  "model_name": "mistral",
  "provider": "local",
  "name": "Ma conversation",
  "temperature": 0.7,
  "message_max": 10,
  "created_at": "2025-12-24T10:30:00Z"
}
```

**Erreurs possibles:**
- `404` : Conversation non trouvée

---

### Supprimer une conversation

#### `DELETE /conversations/{id}`
Supprime une conversation spécifique et tous ses messages.

**Paramètres:**
- `id` (path, integer) : ID de la conversation

**Réponse 200:**
```json
{
  "message": "Conversation 1 deleted"
}
```

**Erreurs possibles:**
- `404` : Conversation non trouvée

---

### Supprimer toutes les conversations

#### `DELETE /conversations`
Supprime toutes les conversations et leurs messages.

**Réponse 200:**
```json
{
  "message": "5 conversation(s) deleted",
  "count": 5
}
```

---

## 💌 Messages

### Envoyer un message

#### `POST /conversations/{id}/message`
Envoie un message dans une conversation et reçoit la réponse du LLM.

**Rate Limit:** 30 requêtes/minute

**Paramètres:**
- `id` (path, integer) : ID de la conversation
- `provider` (query, string, optionnel) : Surcharge du provider pour ce message uniquement (`local` ou `groq`)

**Body (JSON):**
```json
{
  "content": "Bonjour, comment ça va ?"  // REQUIS
}
```

**Exemples d'URL:**
- `POST /conversations/1/message` (utilise le provider de la conversation)
- `POST /conversations/1/message?provider=groq` (force l'utilisation de Groq)

**Réponse 200:**
```json
{
  "id": 42,
  "conversation_id": 1,
  "role": "assistant",
  "content": "Bonjour ! Je vais bien, merci. Comment puis-je vous aider ?",
  "created_at": "2025-12-24T10:35:00Z",
  "metadata": {
    "model": "mistral",
    "tokens": 156
  }
}
```

**Erreurs possibles:**
- `400` : Erreur de validation (contenu manquant ou provider invalide)
- `404` : Conversation non trouvée
- `500` : Erreur lors de la génération de la réponse

---

### Lister les messages d'une conversation

#### `GET /conversations/{id}/message`
Récupère tous les messages d'une conversation.

**Paramètres:**
- `id` (path, integer) : ID de la conversation

**Réponse 200:**
```json
[
  {
    "id": 1,
    "conversation_id": 1,
    "role": "user",
    "content": "Bonjour",
    "created_at": "2025-12-24T10:30:00Z"
  },
  {
    "id": 2,
    "conversation_id": 1,
    "role": "assistant",
    "content": "Bonjour ! Comment puis-je vous aider ?",
    "created_at": "2025-12-24T10:30:05Z"
  }
]
```

---

### Récupérer un message spécifique

#### `GET /conversations/{id}/message/{message_id}`
Récupère un message spécifique.

**Paramètres:**
- `id` (path, integer) : ID de la conversation
- `message_id` (path, integer) : ID du message

**Réponse 200:**
```json
{
  "id": 2,
  "conversation_id": 1,
  "role": "assistant",
  "content": "Bonjour ! Comment puis-je vous aider ?",
  "created_at": "2025-12-24T10:30:05Z"
}
```

**Erreurs possibles:**
- `404` : Message non trouvé

---

### Supprimer un message

#### `DELETE /conversations/{id}/message/{message_id}`
Supprime un message spécifique.

**Paramètres:**
- `id` (path, integer) : ID de la conversation
- `message_id` (path, integer) : ID du message

**Réponse 200:**
```json
{
  "message": "Message 2 deleted"
}
```

**Erreurs possibles:**
- `404` : Message non trouvé

---

## 🔌 Providers

### Lister tous les providers

#### `GET /providers`
Liste tous les providers disponibles et leur statut.

**Réponse 200:**
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

---

### Lister les modèles d'un provider

#### `GET /providers/{provider_name}/models`
Récupère la liste des modèles disponibles pour un provider.

**Paramètres:**
- `provider_name` (path, string) : Nom du provider (`local` ou `groq`)

**Réponse 200 (local):**
```json
{
  "provider": "local",
  "models": [
    {
      "identifier": "mistral",
      "name": "Mistral 7B",
      "size": "7B"
    },
    {
      "identifier": "qwen",
      "name": "Qwen 2.5",
      "size": "14B"
    }
  ],
  "count": 2
}
```

**Réponse 200 (groq):**
```json
{
  "provider": "groq",
  "models": [
    {
      "identifier": "llama-3.1-70b-versatile",
      "name": "Llama 3.1 70B",
      "owned_by": "meta"
    }
  ],
  "count": 1
}
```

**Erreurs possibles:**
- `400` : Provider invalide (doit être 'local' ou 'groq')
- `503` : Provider non disponible ou erreur de connexion

---

## 🧠 RAG - Retrieval-Augmented Generation

Le système RAG permet d'enrichir les réponses des LLM avec des connaissances issues de documents externes.

### Health Check RAG

#### `GET /rag/health`
Vérifie l'état du système RAG.

**Réponse 200:**
```json
{
  "status": "ok",
  "rag_system": "active",
  "collection_info": {
    "collection_name": "rag_collection",
    "total_chunks": 245,
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "persist_directory": "./data/rag_db"
  }
}
```

**Erreurs possibles:**
- `503` : RAGManager non initialisé

---

### Formats supportés

#### `GET /rag/formats`
Liste les formats de fichiers supportés pour l'import de documents.

**Réponse 200:**
```json
{
  "status": "success",
  "supported_formats": [".docx", ".md", ".pdf", ".txt"],
  "count": 4
}
```

---

### Ajouter un document (par chemin)

#### `POST /rag/documents`
Ajoute un document à la base de connaissances.

**Option 1 - Via chemin de fichier:**

**Body (JSON):**
```json
{
  "file_path": "/path/to/document.pdf",  // REQUIS
  "metadata": {                          // Optionnel
    "author": "John Doe",
    "category": "technical"
  },
  "document_id": "custom-doc-123"        // Optionnel (auto-généré sinon)
}
```

**Option 2 - Via texte direct:**

**Body (JSON):**
```json
{
  "text": "Contenu du document...",     // REQUIS
  "source": "manual_input",             // Optionnel
  "metadata": {                         // Optionnel
    "topic": "AI"
  },
  "document_id": "custom-text-456"      // Optionnel
}
```

**Réponse 200:**
```json
{
  "status": "success",
  "message": "Document ajouté avec succès",
  "file_path": "/path/to/document.pdf",
  "chunks_added": 12
}
```

**Erreurs possibles:**
- `400` : Body invalide, format non supporté, ou texte vide
- `403` : Permission refusée pour accéder au fichier
- `404` : Fichier non trouvé
- `503` : RAGManager non initialisé

---

### Upload un document (multipart)

#### `POST /rag/documents/upload`
Upload un fichier et l'ajoute automatiquement à la base de connaissances.

**Content-Type:** `multipart/form-data`

**Form Data:**
- `file` (file, REQUIS) : Fichier à uploader
- `metadata` (string, optionnel) : JSON string avec métadonnées
- `document_id` (string, optionnel) : ID personnalisé

**Exemple avec curl:**
```bash
curl -X POST http://localhost:5001/rag/documents/upload \
  -F "file=@document.pdf" \
  -F 'metadata={"author":"Jane Doe"}' \
  -F "document_id=my-doc-123"
```

**Réponse 200:**
```json
{
  "status": "success",
  "message": "Fichier uploadé et ajouté avec succès",
  "filename": "document.pdf",
  "chunks_added": 15
}
```

**Erreurs possibles:**
- `400` : Aucun fichier fourni, fichier vide, ou format non supporté
- `503` : RAGManager non initialisé

---

### Rechercher des documents

#### `POST /rag/search`
Recherche les documents les plus pertinents pour une requête.

**Body (JSON):**
```json
{
  "query": "Comment fonctionne l'IA ?",  // REQUIS
  "n_results": 5,                        // Optionnel (défaut: 5)
  "filter_metadata": {                   // Optionnel
    "category": "technical"
  }
}
```

**Réponse 200:**
```json
{
  "status": "success",
  "query": "Comment fonctionne l'IA ?",
  "results_count": 3,
  "results": [
    {
      "document": "L'intelligence artificielle est...",
      "metadata": {
        "source": "/docs/ai.pdf",
        "file_name": "ai.pdf",
        "chunk_index": 2,
        "total_chunks": 10,
        "category": "technical"
      },
      "distance": 0.234
    },
    {
      "document": "Les réseaux de neurones sont...",
      "metadata": {
        "source": "/docs/neural.pdf",
        "chunk_index": 0,
        "total_chunks": 5
      },
      "distance": 0.456
    }
  ]
}
```

**Note:** Plus la `distance` est faible, plus le document est pertinent.

**Erreurs possibles:**
- `400` : Query manquante ou n_results invalide
- `503` : RAGManager non initialisé

---

### Récupérer le contexte formaté

#### `POST /rag/context`
Récupère le contexte formaté pour injection dans un prompt LLM.

**Body (JSON):**
```json
{
  "query": "Explique-moi le RAG",
  "n_results": 3,
  "filter_metadata": {
    "topic": "RAG"
  }
}
```

**Réponse 200:**
```json
{
  "status": "success",
  "query": "Explique-moi le RAG",
  "context": "[Document 1 - Source: rag.pdf, Chunk 0]\nLe RAG (Retrieval-Augmented Generation) est...\n\n[Document 2 - Source: embeddings.pdf, Chunk 3]\nLes embeddings permettent de...",
  "context_length": 1234
}
```

**Usage recommandé:** Injecter le `context` dans le system prompt ou avec le message utilisateur.

**Erreurs possibles:**
- `400` : Query manquante
- `503` : RAGManager non initialisé

---

### Informations sur la collection

#### `GET /rag/collection/info`
Récupère les informations sur la collection RAG actuelle.

**Réponse 200:**
```json
{
  "status": "success",
  "collection_info": {
    "collection_name": "rag_collection",
    "total_chunks": 245,
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "persist_directory": "./data/rag_db"
  }
}
```

---

### Supprimer la collection

#### `DELETE /rag/collection/delete`
Supprime toute la collection RAG (tous les documents et chunks).

**⚠️ ATTENTION:** Cette action est irréversible !

**Réponse 200:**
```json
{
  "status": "success",
  "message": "Collection supprimée avec succès"
}
```

---

## ❌ Codes d'erreur

| Code | Signification | Description |
|------|---------------|-------------|
| `200` | OK | Requête réussie |
| `201` | Created | Ressource créée avec succès |
| `400` | Bad Request | Erreur de validation des paramètres |
| `403` | Forbidden | Permission refusée |
| `404` | Not Found | Ressource non trouvée |
| `429` | Too Many Requests | Rate limit dépassé |
| `500` | Internal Server Error | Erreur serveur interne |
| `503` | Service Unavailable | Service temporairement indisponible |

### Format des erreurs

Toutes les erreurs retournent un JSON avec ce format :

```json
{
  "error": "Message d'erreur principal",
  "details": ["Détail 1", "Détail 2"]  // Optionnel
}
```

ou

```json
{
  "status": "error",
  "message": "Description de l'erreur"
}
```

---

## 🚀 Exemples d'utilisation

### Scénario 1 : Conversation simple

```python
import requests

BASE_URL = "http://localhost:5001"

# 1. Créer une conversation
conversation = requests.post(f"{BASE_URL}/conversations", json={
    "model_name": "mistral",
    "provider": "local",
    "name": "Test API"
}).json()

conv_id = conversation["id"]

# 2. Envoyer un message
response = requests.post(
    f"{BASE_URL}/conversations/{conv_id}/message",
    json={"content": "Bonjour !"}
).json()

print(response["content"])  # Affiche la réponse du LLM
```

---

### Scénario 2 : Utiliser le RAG

```python
import requests

BASE_URL = "http://localhost:5001"

# 1. Upload un document
with open("documentation.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        f"{BASE_URL}/rag/documents/upload",
        files=files
    ).json()

print(f"Document ajouté: {response['chunks_added']} chunks")

# 2. Rechercher dans les documents
search_result = requests.post(f"{BASE_URL}/rag/search", json={
    "query": "Comment installer l'application ?",
    "n_results": 3
}).json()

# 3. Récupérer le contexte formaté
context_result = requests.post(f"{BASE_URL}/rag/context", json={
    "query": "Comment installer l'application ?",
    "n_results": 3
}).json()

context = context_result["context"]

# 4. Créer une conversation avec le contexte
conversation = requests.post(f"{BASE_URL}/conversations", json={
    "model_name": "mistral",
    "system_prompt": f"Utilise ce contexte pour répondre:\n\n{context}"
}).json()

# 5. Poser la question
response = requests.post(
    f"{BASE_URL}/conversations/{conversation['id']}/message",
    json={"content": "Comment installer l'application ?"}
).json()

print(response["content"])
```

---

### Scénario 3 : Utiliser plusieurs providers

```python
import requests

BASE_URL = "http://localhost:5001"

# 1. Lister les providers disponibles
providers = requests.get(f"{BASE_URL}/providers").json()
print("Providers:", providers)

# 2. Créer une conversation avec Groq
conversation = requests.post(f"{BASE_URL}/conversations", json={
    "model_name": "llama-3.1-70b-versatile",
    "provider": "groq"
}).json()

conv_id = conversation["id"]

# 3. Envoyer un message (utilise Groq par défaut)
response = requests.post(
    f"{BASE_URL}/conversations/{conv_id}/message",
    json={"content": "Résume-moi l'actualité IA"}
).json()

# 4. Forcer l'utilisation du provider local pour un message
response_local = requests.post(
    f"{BASE_URL}/conversations/{conv_id}/message?provider=local",
    json={"content": "Même question"}
).json()
```

---

### Scénario 4 : Ajouter du texte au RAG

```python
import requests

BASE_URL = "http://localhost:5001"

# Ajouter du texte directement
response = requests.post(f"{BASE_URL}/rag/documents", json={
    "text": """
    Le RAG (Retrieval-Augmented Generation) combine la recherche 
    d'informations avec la génération de texte. Il permet aux LLM 
    d'accéder à des connaissances externes.
    """,
    "source": "documentation_interne",
    "metadata": {
        "topic": "RAG",
        "author": "Tech Team"
    }
}).json()

print(f"Texte ajouté: {response['chunks_added']} chunks")

# Rechercher dans les documents
search = requests.post(f"{BASE_URL}/rag/search", json={
    "query": "Qu'est-ce que le RAG ?",
    "filter_metadata": {"topic": "RAG"}
}).json()

print(f"Résultats trouvés: {search['results_count']}")
```

---

## 🛠️ Configuration

### Variables d'environnement

#### RAG
- `RAG_COLLECTION_NAME` : Nom de la collection (défaut: `rag_collection`)
- `RAG_PERSIST_DIR` : Répertoire de persistance (défaut: `./data/rag_db`)
- `RAG_EMBEDDING_MODEL` : Modèle d'embeddings (défaut: `all-MiniLM-L6-v2`)
- `RAG_CHUNK_SIZE` : Taille des chunks (défaut: `1000`)
- `RAG_CHUNK_OVERLAP` : Chevauchement (défaut: `200`)
- `RAG_DEVICE` : Device pour embeddings (`cpu`, `cuda`, ou `None` pour auto)

#### CORS & Rate Limiting
- `CORS_ORIGINS` : Origins autorisées (défaut: `*`)
- `RATELIMIT_STORAGE_URL` : URL de stockage rate limit (défaut: `memory://`)

#### Providers
- `GROQ_API_KEY` : Clé API pour Groq (requis pour utiliser Groq)
- Configuration locale : Voir documentation spécifique du provider local

---

## 📊 Limites et performances

- **Taille max message** : Dépend du modèle utilisé
- **Taille max fichier upload** : Configurable (défaut Flask)
- **Formats RAG** : `.txt`, `.md`, `.pdf`, `.docx`
- **Modèle embeddings** : Peut être changé (performances variables)
- **Chunk size optimal** : 500-2000 caractères selon le cas d'usage

---

## 🆘 Support & Troubleshooting

### Le RAG n'est pas disponible
- Vérifiez que ChromaDB et sentence-transformers sont installés
- Consultez les logs au démarrage de l'application

### Provider local non disponible
- Vérifiez que l'API de base est démarrée et accessible
- Testez l'endpoint `GET /` pour voir le statut

### Rate limit dépassé
- Attendez la fenêtre de temps appropriée
- Ou configurez `RATELIMIT_STORAGE_URL` pour une limite persistante

### Erreur 500 récurrente
- Consultez les logs serveur
- Vérifiez la configuration des modèles et providers

---

## 📝 Notes

- Les conversations et messages sont persistés en base de données
- Le système RAG utilise ChromaDB pour le stockage vectoriel
- Les embeddings sont calculés avec sentence-transformers
- L'API supporte le streaming pour certains providers (voir doc spécifique)

---

**Dernière mise à jour:** 24 décembre 2025
