# Documentation complète de l'API RAG

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture et fonctionnement](#architecture-et-fonctionnement)
3. [Configuration](#configuration)
4. [Endpoints disponibles](#endpoints-disponibles)
5. [Formats de fichiers supportés](#formats-de-fichiers-supportés)
6. [Gestion d'erreurs](#gestion-derreurs)
7. [Exemples d'utilisation](#exemples-dutilisation)

---

## Vue d'ensemble

L'API RAG (Retrieval-Augmented Generation) permet de :
- **Indexer** des documents dans une base de données vectorielle (ChromaDB)
- **Rechercher** des documents pertinents par similarité sémantique
- **Récupérer** du contexte formaté pour l'injection dans des prompts LLM

### Technologies utilisées

- **Backend** : Flask (Python)
- **Base de données vectorielle** : ChromaDB
- **Modèle d'embedding** : Sentence Transformers (par défaut: `all-MiniLM-L6-v2`)
- **Découpage de texte** : LangChain RecursiveCharacterTextSplitter

### Base URL

Toutes les routes RAG sont préfixées par `/rag` :
```
http://localhost:5001/rag
```

---

## Architecture et fonctionnement

### Flux de traitement

1. **Ajout de document** :
   ```
   Document → Chargement → Découpage en chunks → Génération d'embeddings → Stockage dans ChromaDB
   ```

2. **Recherche** :
   ```
   Requête → Embedding de la requête → Recherche par similarité → Retour des chunks pertinents
   ```

### Découpage des documents

- **Taille des chunks** : 1000 caractères par défaut (configurable)
- **Chevauchement** : 200 caractères par défaut (configurable)
- Chaque chunk est stocké avec ses métadonnées (source, index, etc.)

### Métadonnées automatiques

Chaque chunk stocké contient automatiquement :
- `source` : Chemin du fichier source ou source du texte
- `file_name` : Nom du fichier (si applicable)
- `chunk_index` : Index du chunk dans le document (0-based)
- `total_chunks` : Nombre total de chunks du document

Vous pouvez ajouter des métadonnées personnalisées lors de l'ajout de documents.

---

## Configuration

### Variables d'environnement

L'API peut être configurée via des variables d'environnement :

| Variable | Description | Défaut |
|----------|-------------|--------|
| `RAG_COLLECTION_NAME` | Nom de la collection ChromaDB | `rag_collection` |
| `RAG_PERSIST_DIR` | Répertoire de persistance de la base | `./data/rag_db` |
| `RAG_EMBEDDING_MODEL` | Modèle d'embedding Sentence Transformers | `all-MiniLM-L6-v2` |
| `RAG_CHUNK_SIZE` | Taille des chunks en caractères | `1000` |
| `RAG_CHUNK_OVERLAP` | Chevauchement entre chunks | `200` |
| `RAG_DEVICE` | Device pour les embeddings (`cuda`, `cpu`, ou `None` pour auto) | `None` |

### Exemple de configuration

```bash
export RAG_COLLECTION_NAME="ma_collection"
export RAG_PERSIST_DIR="./data/ma_base_rag"
export RAG_EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
export RAG_CHUNK_SIZE=1500
export RAG_CHUNK_OVERLAP=300
export RAG_DEVICE="cuda"
```

---

## Endpoints disponibles

### 1. Health Check

Vérifie l'état du système RAG et retourne les informations de la collection.

**Endpoint** : `GET /rag/health`

**Paramètres** : Aucun

**Réponse réussie** (200) :
```json
{
  "status": "ok",
  "rag_system": "active",
  "collection_info": {
    "collection_name": "rag_collection",
    "total_chunks": 42,
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "persist_directory": "./data/rag_db"
  }
}
```

**Réponse d'erreur** (503) :
```json
{
  "status": "error",
  "message": "RAGManager non initialisé"
}
```

**Réponse d'erreur** (500) :
```json
{
  "status": "error",
  "message": "Message d'erreur détaillé"
}
```

---

### 2. Formats supportés

Liste tous les formats de fichiers supportés par l'API.

**Endpoint** : `GET /rag/formats`

**Paramètres** : Aucun

**Réponse réussie** (200) :
```json
{
  "status": "success",
  "supported_formats": [".docx", ".md", ".pdf", ".txt"],
  "count": 4
}
```

---

### 3. Ajouter un document

Ajoute un document à la base de connaissances via un chemin de fichier ou du texte direct.

**Endpoint** : `POST /rag/documents`

**Content-Type** : `application/json`

**Body JSON** :

**Option 1 : Via chemin de fichier**
```json
{
  "file_path": "/chemin/vers/document.pdf",
  "metadata": {
    "category": "technique",
    "author": "John Doe",
    "date": "2024-01-15"
  },
  "document_id": "doc_123"
}
```

**Option 2 : Via texte direct**
```json
{
  "text": "Contenu du document à indexer...",
  "source": "manual_input",
  "metadata": {
    "category": "note",
    "priority": "high"
  },
  "document_id": "text_456"
}
```

**Paramètres** :

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `file_path` | string | Oui* | Chemin vers le fichier à indexer |
| `text` | string | Oui* | Texte à indexer directement |
| `source` | string | Non | Source du texte (défaut: `manual_input`) |
| `metadata` | object | Non | Métadonnées personnalisées à associer |
| `document_id` | string | Non | ID unique du document (généré automatiquement si absent) |

\* L'un des deux (`file_path` ou `text`) est requis.

**Réponse réussie** (200) :

Via `file_path` :
```json
{
  "status": "success",
  "message": "Document ajouté avec succès",
  "file_path": "/chemin/vers/document.pdf",
  "chunks_added": 15
}
```

Via `text` :
```json
{
  "status": "success",
  "message": "Texte ajouté avec succès",
  "chunks_added": 3
}
```

**Réponses d'erreur** :

- **400** : Format non supporté, body invalide, ou texte vide
- **403** : Permission refusée pour accéder au fichier
- **404** : Fichier non trouvé
- **500** : Erreur interne
- **503** : RAGManager non initialisé

**Exemple d'erreur** (400) :
```json
{
  "status": "error",
  "message": "Format de fichier non supporté: .png. Formats supportés: .docx, .md, .pdf, .txt"
}
```

---

### 4. Upload de fichier

Upload un fichier via form-data et l'ajoute à la base de connaissances.

**Endpoint** : `POST /rag/documents/upload`

**Content-Type** : `multipart/form-data`

**Paramètres form-data** :

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `file` | file | Oui | Fichier à uploader |
| `metadata` | string (JSON) | Non | Métadonnées au format JSON string |
| `document_id` | string | Non | ID unique du document |

**Réponse réussie** (200) :
```json
{
  "status": "success",
  "message": "Fichier uploadé et ajouté avec succès",
  "filename": "document.pdf",
  "chunks_added": 12
}
```

**Réponses d'erreur** :

- **400** : Aucun fichier fourni, format non supporté, ou metadata JSON invalide
- **404** : Fichier non trouvé (après upload)
- **500** : Erreur interne
- **503** : RAGManager non initialisé

**Note** : Le fichier est temporairement sauvegardé, traité, puis supprimé automatiquement.

---

### 5. Recherche

Effectue une recherche sémantique de documents pertinents pour une requête.

**Endpoint** : `POST /rag/search`

**Content-Type** : `application/json`

**Body JSON** :
```json
{
  "query": "Qu'est-ce que le machine learning?",
  "n_results": 5,
  "filter_metadata": {
    "category": "technique"
  }
}
```

**Paramètres** :

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `query` | string | Oui | Requête de recherche |
| `n_results` | integer | Non | Nombre de résultats à retourner (défaut: 5) |
| `filter_metadata` | object | Non | Filtres sur les métadonnées (ex: `{"source": "file.pdf"}`) |

**Réponse réussie** (200) :
```json
{
  "status": "success",
  "query": "Qu'est-ce que le machine learning?",
  "results_count": 5,
  "results": [
    {
      "document": "Le machine learning est une branche de l'intelligence artificielle...",
      "metadata": {
        "source": "/path/to/document.pdf",
        "file_name": "document.pdf",
        "chunk_index": 2,
        "total_chunks": 15,
        "category": "technique"
      },
      "distance": 0.234
    },
    {
      "document": "Les algorithmes de machine learning apprennent à partir de données...",
      "metadata": {
        "source": "/path/to/autre_doc.pdf",
        "file_name": "autre_doc.pdf",
        "chunk_index": 0,
        "total_chunks": 8
      },
      "distance": 0.312
    }
  ]
}
```

**Champs des résultats** :

- `document` : Texte du chunk trouvé
- `metadata` : Métadonnées associées au chunk
- `distance` : Distance de similarité (plus petit = plus similaire, 0 = identique)

**Réponses d'erreur** :

- **400** : Body JSON requis, champ `query` manquant ou invalide, `n_results` invalide
- **500** : Erreur interne
- **503** : RAGManager non initialisé

---

### 6. Contexte formaté

Récupère le contexte formaté pour une requête, prêt à être injecté dans un prompt LLM.

**Endpoint** : `POST /rag/context`

**Content-Type** : `application/json`

**Body JSON** :
```json
{
  "query": "Explique-moi le deep learning",
  "n_results": 3,
  "filter_metadata": {
    "category": "technique"
  }
}
```

**Paramètres** :

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `query` | string | Oui | Requête de recherche |
| `n_results` | integer | Non | Nombre de résultats à inclure (défaut: 5) |
| `filter_metadata` | object | Non | Filtres sur les métadonnées |

**Réponse réussie** (200) :
```json
{
  "status": "success",
  "query": "Explique-moi le deep learning",
  "context": "[Document 1 - Source: /path/to/doc1.pdf, Chunk 5]\nLe deep learning utilise des réseaux de neurones...\n\n[Document 2 - Source: /path/to/doc2.pdf, Chunk 0]\nLes réseaux de neurones profonds sont composés...",
  "context_length": 1250
}
```

**Format du contexte** :

Le contexte est formaté comme suit :
```
[Document N - Source: <source>, Chunk <index>]
<contenu du chunk>
```

**Réponses d'erreur** :

- **400** : Body JSON requis, champ `query` manquant ou invalide
- **500** : Erreur interne
- **503** : RAGManager non initialisé

---

### 7. Informations de la collection

Retourne des informations détaillées sur la collection actuelle.

**Endpoint** : `GET /rag/collection/info`

**Paramètres** : Aucun

**Réponse réussie** (200) :
```json
{
  "status": "success",
  "collection_info": {
    "collection_name": "rag_collection",
    "total_chunks": 127,
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "persist_directory": "./data/rag_db"
  }
}
```

**Champs retournés** :

- `collection_name` : Nom de la collection ChromaDB
- `total_chunks` : Nombre total de chunks indexés
- `embedding_model` : Modèle d'embedding utilisé
- `chunk_size` : Taille des chunks configurée
- `chunk_overlap` : Chevauchement entre chunks configuré
- `persist_directory` : Répertoire de persistance (ou `null` si en mémoire)

**Réponses d'erreur** :

- **500** : Erreur interne
- **503** : RAGManager non initialisé

---

### 8. Supprimer la collection

Supprime complètement la collection actuelle et toutes ses données.

**Endpoint** : `DELETE /rag/collection/delete`

**Paramètres** : Aucun

**Réponse réussie** (200) :
```json
{
  "status": "success",
  "message": "Collection supprimée avec succès"
}
```

**⚠️ Attention** : Cette opération est irréversible. Tous les documents indexés seront perdus.

**Réponses d'erreur** :

- **500** : Erreur interne
- **503** : RAGManager non initialisé

---

## Formats de fichiers supportés

L'API supporte actuellement les formats suivants :

| Format | Extension | Description |
|--------|-----------|-------------|
| Texte | `.txt` | Fichiers texte brut (UTF-8) |
| Markdown | `.md` | Fichiers Markdown |
| PDF | `.pdf` | Documents PDF (extraction de texte) |
| Word | `.docx` | Documents Microsoft Word |

### Limitations

- **PDF** : Extraction de texte uniquement (pas d'images, tableaux complexes, etc.)
- **DOCX** : Extraction du texte des paragraphes uniquement (pas de tableaux, images, etc.)
- **Encodage** : Les fichiers texte doivent être en UTF-8

---

## Gestion d'erreurs

### Codes HTTP

| Code | Signification | Quand |
|------|---------------|-------|
| 200 | Succès | Opération réussie |
| 400 | Requête invalide | Paramètres manquants, format invalide, etc. |
| 403 | Permission refusée | Accès au fichier refusé |
| 404 | Non trouvé | Fichier inexistant, route non trouvée |
| 500 | Erreur interne | Erreur serveur inattendue |
| 503 | Service indisponible | RAGManager non initialisé |

### Format des réponses d'erreur

Toutes les erreurs suivent le format :
```json
{
  "status": "error",
  "message": "Description de l'erreur"
}
```

### Types d'erreurs gérées

#### Fichiers inexistants
- **Erreur** : `FileNotFoundError`
- **Code** : 404
- **Message** : `"Le fichier n'existe pas: /chemin/fichier.pdf"`

#### Formats non supportés
- **Erreur** : `ValueError`
- **Code** : 400
- **Message** : `"Format de fichier non supporté: .png. Formats supportés: .docx, .md, .pdf, .txt"`

#### Fichiers invalides
- **PDF corrompu** : `ValueError` avec message descriptif
- **Encodage invalide** : `ValueError` avec suggestion UTF-8
- **Permission refusée** : `PermissionError` avec code 403

#### Erreurs ChromaDB
- **Erreur** : `RuntimeError`
- **Code** : 500
- **Message** : Description de l'erreur ChromaDB

#### Validation des requêtes
- **Body JSON manquant** : 400
- **Champs requis manquants** : 400
- **Types de données invalides** : 400

---

## Exemples d'utilisation

### Exemple 1 : Ajouter un document PDF

```bash
curl -X POST http://localhost:5001/rag/documents \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/chemin/vers/document.pdf",
    "metadata": {
      "category": "technique",
      "author": "John Doe"
    }
  }'
```

**Réponse** :
```json
{
  "status": "success",
  "message": "Document ajouté avec succès",
  "file_path": "/chemin/vers/document.pdf",
  "chunks_added": 15
}
```

### Exemple 2 : Ajouter du texte direct

```bash
curl -X POST http://localhost:5001/rag/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Le machine learning est une méthode d'analyse de données qui automatise la construction de modèles analytiques.",
    "source": "manuel",
    "metadata": {
      "type": "définition"
    }
  }'
```

### Exemple 3 : Upload de fichier

```bash
curl -X POST http://localhost:5001/rag/documents/upload \
  -F "file=@document.pdf" \
  -F "metadata={\"category\": \"technique\", \"priority\": \"high\"}" \
  -F "document_id=doc_001"
```

### Exemple 4 : Recherche simple

```bash
curl -X POST http://localhost:5001/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qu'est-ce que le machine learning?",
    "n_results": 5
  }'
```

### Exemple 5 : Recherche avec filtres

```bash
curl -X POST http://localhost:5001/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "algorithmes d'apprentissage",
    "n_results": 3,
    "filter_metadata": {
      "category": "technique"
    }
  }'
```

### Exemple 6 : Récupérer le contexte formaté

```bash
curl -X POST http://localhost:5001/rag/context \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explique le deep learning",
    "n_results": 3
  }'
```

**Réponse** :
```json
{
  "status": "success",
  "query": "Explique le deep learning",
  "context": "[Document 1 - Source: /path/to/doc1.pdf, Chunk 5]\nLe deep learning utilise des réseaux de neurones artificiels avec plusieurs couches...\n\n[Document 2 - Source: /path/to/doc2.pdf, Chunk 0]\nLes réseaux de neurones profonds permettent d'apprendre des représentations complexes...",
  "context_length": 1250
}
```

### Exemple 7 : Vérifier l'état du système

```bash
curl -X GET http://localhost:5001/rag/health
```

### Exemple 8 : Obtenir les informations de la collection

```bash
curl -X GET http://localhost:5001/rag/collection/info
```

### Exemple 9 : Supprimer la collection

```bash
curl -X DELETE http://localhost:5001/rag/collection/delete
```

---

## Notes importantes

### Performance

- Le chargement initial du modèle d'embedding peut prendre quelques secondes
- Les recherches sont rapides une fois le modèle chargé
- L'ajout de documents volumineux peut prendre du temps selon la taille

### Persistance

- Par défaut, les données sont persistées dans `./data/rag_db`
- La collection est réutilisée au redémarrage de l'API
- Pour repartir de zéro, supprimez la collection ou changez le nom de collection

### Métadonnées et filtres

- Les métadonnées sont stockées avec chaque chunk
- Les filtres de métadonnées utilisent la syntaxe ChromaDB `where`
- Exemple de filtre complexe : `{"source": {"$contains": "pdf"}}`

### IDs de documents

- Si non fourni, l'ID est généré automatiquement (hash MD5 du chemin ou du texte)
- Les IDs doivent être uniques par document
- Les chunks d'un même document partagent le même préfixe d'ID

---

## Support et dépannage

### Vérifier que l'API fonctionne

```bash
curl http://localhost:5001/
```

### Vérifier le système RAG

```bash
curl http://localhost:5001/rag/health
```

### Vérifier les formats supportés

```bash
curl http://localhost:5001/rag/formats
```

### Logs

Les erreurs sont loggées avec le module `logging` de Python. Vérifiez les logs de l'application pour plus de détails sur les erreurs.
