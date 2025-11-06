# Documentation des Routes RAG API

## Routes principales implémentées

### 1. Health Check
**GET** `/rag/health`
- Vérifie l'état du système RAG
- Retourne les informations de la collection

### 2. Formats supportés
**GET** `/rag/formats`
- Liste tous les formats de fichiers supportés (.txt, .md, .pdf, .docx)

### 3. Ajouter un document
**POST** `/rag/documents`
- Ajoute un document via `file_path` ou texte direct
- Body JSON avec `file_path` OU `text`
- Optionnel: `metadata`, `document_id`

### 4. Upload de fichier
**POST** `/rag/documents/upload`
- Upload un fichier via form-data
- Fichier temporairement sauvegardé puis supprimé
- Optionnel: `metadata` (JSON), `document_id`

### 5. Recherche
**POST** `/rag/search`
- Recherche sémantique de documents pertinents
- Body JSON: `query`, `n_results` (optionnel), `filter_metadata` (optionnel)
- Retourne les résultats avec métadonnées et distances

### 6. Contexte formaté
**POST** `/rag/context`
- Récupère le contexte formaté pour injection dans un prompt
- Body JSON: `query`, `n_results` (optionnel), `filter_metadata` (optionnel)
- Retourne le contexte prêt à utiliser

### 7. Informations collection
**GET** `/rag/collection/info`
- Retourne les informations sur la collection actuelle
- Nombre de chunks, modèle d'embedding, configuration

### 8. Supprimer collection
**DELETE** `/rag/collection/delete`
- Supprime complètement la collection actuelle

---

## Routes de monitoring recommandées (à implémenter)

### Monitoring système
- **GET** `/rag/stats` - Statistiques détaillées (nombre de documents, taille, etc.)
- **GET** `/rag/documents/list` - Liste tous les documents indexés
- **GET** `/rag/documents/<document_id>` - Informations sur un document spécifique
- **GET** `/rag/documents/<document_id>/chunks` - Liste les chunks d'un document

### Monitoring performance
- **GET** `/rag/metrics` - Métriques de performance (temps de recherche, taille embeddings, etc.)
- **GET** `/rag/embedding/model/info` - Informations sur le modèle d'embedding utilisé

### Monitoring qualité
- **POST** `/rag/search/test` - Test de recherche avec métriques de qualité
- **GET** `/rag/collection/stats` - Statistiques avancées (distribution des chunks, sources, etc.)

---

## Gestion d'erreurs implémentée

### Dans `rag.py` :

1. **Fichiers inexistants** :
   - Vérification avec `Path.exists()` avant chargement
   - Exception `FileNotFoundError` avec message clair

2. **Formats non supportés** :
   - Liste des formats supportés dans `DocumentLoader.SUPPORTED_FORMATS`
   - Méthode `is_supported_format()` pour vérification
   - Exception `ValueError` avec liste des formats supportés
   - Exemple: `.png` → erreur claire avec formats acceptés

3. **Fichiers invalides** :
   - Vérification que c'est un fichier (pas un répertoire)
   - Gestion des PDF corrompus (`PdfReadError`)
   - Gestion des erreurs d'encodage (`UnicodeDecodeError`)
   - Gestion des permissions (`PermissionError`)

4. **Erreurs ChromaDB** :
   - Try/catch autour des opérations ChromaDB
   - Exception `RuntimeError` avec message descriptif

5. **Erreurs globales** :
   - Logging de toutes les erreurs
   - Exceptions typées et spécifiques
   - Messages d'erreur clairs et actionnables

### Dans les routes API :

1. **Validation des requêtes** :
   - Vérification de la présence des champs requis
   - Validation des types de données
   - Codes HTTP appropriés (400, 404, 403, 500)

2. **Gestion des erreurs** :
   - Try/catch dans chaque route
   - Messages d'erreur JSON structurés
   - Logging des erreurs pour debugging

3. **Gestionnaires globaux** :
   - `@app.errorhandler(404)` pour routes non trouvées
   - `@app.errorhandler(500)` pour erreurs internes

---

## Exemples d'utilisation

### Ajouter un document
```bash
curl -X POST http://localhost:5000/rag/documents \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "metadata": {"category": "technical"}
  }'
```

### Recherche
```bash
curl -X POST http://localhost:5000/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qu'est-ce que le machine learning?",
    "n_results": 5
  }'
```

### Upload de fichier
```bash
curl -X POST http://localhost:5000/rag/documents/upload \
  -F "file=@document.pdf" \
  -F "metadata={\"category\": \"technical\"}"
```

