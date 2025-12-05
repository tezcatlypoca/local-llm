# Local LLM API - Système d'inférence et de recherche augmentée

## Description courte

Système d'API modulaire en Python/Flask pour l'inférence de modèles LLM locaux et la recherche augmentée par génération (RAG). Architecture en trois couches : Base API pour la gestion GPU et l'inférence, Overlay API pour la gestion du contexte et la persistance, et RAG API pour l'indexation et la recherche sémantique de documents.

---

## Description détaillée

### Vue d'ensemble

Ce projet est une plateforme complète d'inférence LLM locale avec système RAG intégré, conçue avec une architecture modulaire en trois couches distinctes. Chaque couche a une responsabilité spécifique, permettant une séparation claire des préoccupations et une évolutivité optimale.

### Architecture en trois couches

#### 1. Base API - Gestion bas niveau des modèles LLM

La **Base API** (port 5000) constitue la couche fondamentale du système. Elle gère directement l'interaction avec les modèles LLM et les ressources GPU :

- **Gestion du cycle de vie des modèles** : Chargement et déchargement dynamique de modèles sur les GPUs AMD via PyTorch/ROCm
- **Inférence directe** : Exécution des requêtes d'inférence avec gestion optimisée de la mémoire GPU
- **Gestion multi-GPU** : Support de plusieurs GPUs avec allocation intelligente des modèles
- **API REST simple** : Endpoints pour lister les modèles disponibles, charger/décharger des modèles, et effectuer des inférences

Cette couche est conçue pour être légère et performante, se concentrant uniquement sur l'exécution des modèles sans logique métier.

#### 2. Overlay API - Gestion du contexte et persistance

La **Overlay API** (port 8000) ajoute une couche d'abstraction et de gestion métier au-dessus de la Base API :

- **Gestion des conversations** : Création, récupération et suppression de conversations avec persistance SQLite
- **Gestion des messages** : Stockage et récupération de l'historique des messages avec relations conversation-message
- **Gestion intelligente du contexte** : 
  - Formatage automatique des messages selon les templates spécifiques à chaque modèle
  - Troncature intelligente des tokens pour respecter les limites de contexte
  - Mise en cache du contexte formaté pour optimiser les performances
- **Abstraction des providers** : Support de multiples providers (local, Groq, etc.) avec interface unifiée
- **Fonctionnalités avancées** :
  - Rate limiting configurable par endpoint
  - CORS configurable
  - Documentation Swagger/OpenAPI intégrée
  - Gestion des erreurs robuste avec codes HTTP appropriés

Cette couche transforme la Base API en un système conversationnel complet, gérant l'état et le contexte des interactions.

#### 3. RAG API - Recherche augmentée par génération

La **RAG API** (port 5001) implémente un système complet de recherche augmentée par génération :

- **Indexation de documents** : 
  - Support de multiples formats (PDF, TXT, Markdown, DOCX)
  - Découpage intelligent en chunks avec chevauchement configurable
  - Génération d'embeddings avec Sentence Transformers
  - Stockage dans ChromaDB (base de données vectorielle)
- **Recherche sémantique** : 
  - Recherche par similarité vectorielle
  - Filtrage par métadonnées
  - Retour de résultats avec scores de similarité
- **Récupération de contexte formaté** : Génération de contexte prêt à être injecté dans les prompts LLM
- **Gestion de collection** : Création, consultation et suppression de collections de documents

Cette couche permet d'enrichir les réponses des LLM avec des informations provenant d'une base de connaissances indexée.

### Technologies utilisées

- **Backend** : Flask (Python)
- **Base de données relationnelle** : SQLite3 avec pool de connexions
- **Base de données vectorielle** : ChromaDB
- **Embeddings** : Sentence Transformers (modèles configurables)
- **Découpage de texte** : LangChain RecursiveCharacterTextSplitter
- **Inférence LLM** : PyTorch avec support ROCm pour GPUs AMD
- **Documentation** : Swagger/OpenAPI (Flasgger)
- **Sécurité** : Rate limiting, CORS, validation des entrées

### Points techniques remarquables

- **Architecture modulaire** : Séparation claire des responsabilités permettant l'évolution indépendante de chaque couche
- **Optimisation mémoire** : Gestion intelligente du contexte avec troncature et mise en cache
- **Performance** : Pool de connexions SQLite, cache de contexte formaté, recherche vectorielle optimisée
- **Extensibilité** : Système de providers pour ajouter facilement de nouveaux services LLM
- **Robustesse** : Gestion d'erreurs complète, validation des entrées, logging détaillé

### Cas d'usage

- Chatbot conversationnel avec historique persistant
- Assistant IA avec accès à une base de connaissances documentaire
- Système de Q&A sur documents techniques
- Plateforme d'expérimentation avec différents modèles LLM locaux

---

## Stack technique

**Backend** : Python 3.11, Flask, SQLite3, ChromaDB  
**ML/AI** : PyTorch, Sentence Transformers, LangChain  
**Infrastructure** : Docker-ready, Rate limiting, CORS  
**Documentation** : Swagger/OpenAPI

