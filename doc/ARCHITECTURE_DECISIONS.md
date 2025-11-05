# Décisions d'architecture - API Surcouche

## Question : Automatisation du chargement de modèle lors de la création de conversation

### Contexte

Lorsqu'un backend métier souhaite créer une conversation, doit-il :
1. **Option A** : Faire une seule requête qui charge automatiquement le modèle ET crée la conversation
2. **Option B** : Faire plusieurs requêtes séquentielles (charger modèle → créer conversation)

### Analyse des deux approches

#### Option A : Création automatique avec chargement de modèle

**Fonctionnement :**
```
POST /conversations
{
  "model_name": "mistral-7b",
  "auto_load": true  // Optionnel
}
→ Charge le modèle automatiquement
→ Crée la conversation
→ Retourne conversation_id + gpu_id + access_token
```

**Avantages :**
- ✅ Plus simple pour le client (une seule requête)
- ✅ Moins d'appels réseau
- ✅ Apparence plus "magique" et user-friendly

**Inconvénients :**
- ❌ **Violation du principe de responsabilité unique** : La route conversation gère aussi les modèles
- ❌ **Couplage fort** : Conversation et gestion de GPU sont liées
- ❌ **Gestion d'erreurs complexe** : Que faire si le modèle est déjà chargé ? Si le GPU est plein ?
- ❌ **Manque de flexibilité** : Impossible de charger un modèle pour plusieurs conversations
- ❌ **Pas de contrôle** : Le backend métier ne peut pas vérifier l'état avant de créer la conversation
- ❌ **Transparence perdue** : Les données de l'API de base (gpu_id, access_token) doivent être cachées/gérées

#### Option B : Séquence manuelle (recommandée)

**Fonctionnement :**
```
1. POST /models/load/mistral-7b
   → Retourne { gpu_id: 0, access_token: "..." }

2. POST /conversations
   {
     "model_name": "mistral-7b",
     "gpu_id": 0
   }
   → Retourne { conversation_id: "..." }

3. POST /conversations/{id}/messages
   → Envoie un message
```

**Avantages :**
- ✅ **Séparation claire des responsabilités**
  - `/models/*` : Gestion des GPUs et modèles
  - `/conversations/*` : Gestion des conversations
- ✅ **Transparence totale** : Toutes les données de l'API de base sont retournées
- ✅ **Flexibilité maximale** :
  - Charger un modèle pour plusieurs conversations
  - Vérifier l'état avant de créer la conversation
  - Gérer le cycle de vie des modèles indépendamment
- ✅ **Gestion d'erreurs fine** : Chaque étape peut échouer indépendamment
- ✅ **Réutilisabilité** : Un modèle chargé peut servir plusieurs conversations
- ✅ **Respect des principes SOLID** : Single Responsibility, Open/Closed

**Inconvénients :**
- ❌ Plus de requêtes nécessaires (2-3 au lieu de 1)
- ❌ Le backend métier doit gérer la séquence

### Recommandation : Option B (Séquence manuelle)

**Raisons principales :**

1. **Principe de responsabilité unique (SRP)**
   - Chaque route a une responsabilité claire et bien définie
   - Facilite la maintenance et les tests

2. **Transparence de l'API**
   - L'API surcouche doit être transparente et retourner toutes les données de l'API de base
   - Le backend métier peut persister ce qu'il veut selon ses besoins

3. **Flexibilité pour le backend métier**
   - Il peut charger un modèle une fois et créer plusieurs conversations
   - Il peut gérer le cycle de vie des modèles selon sa logique métier
   - Il peut implémenter des stratégies de cache/pooling de modèles

4. **Gestion d'erreurs**
   - Chaque étape peut échouer indépendamment
   - Plus facile de diagnostiquer les problèmes
   - Le backend métier peut gérer les cas d'erreur spécifiques

5. **Évolutivité**
   - Facile d'ajouter de nouvelles fonctionnalités sans casser l'existant
   - Chaque route peut évoluer indépendamment

### Implémentation actuelle

L'implémentation actuelle suit **l'Option B** :

1. **Route `/models/load/<model_name>`** :
   - Retourne toutes les données de l'API de base (gpu_id, access_token, etc.)
   - Enregistre dans le `ModelTracker` (en mémoire, pour faciliter l'utilisation)

2. **Route `/conversations` POST** :
   - Attend un `gpu_id` dans le body
   - Crée la conversation avec le modèle et GPU spécifiés
   - Ne gère PAS le chargement de modèle

3. **Route `/conversations/<id>/messages` POST** :
   - Utilise le `gpu_id` stocké dans la conversation
   - Envoie le message formaté à l'API de base

### Amélioration suggérée : Helper pour le backend métier

Pour faciliter l'utilisation sans violer les principes, on pourrait ajouter :

```python
# Route helper (optionnelle) pour simplifier le workflow
POST /conversations/quick-create
{
  "model_name": "mistral-7b",
  "auto_load": true
}
→ Charge le modèle si nécessaire
→ Crée la conversation
→ Retourne { conversation_id, gpu_id, access_token, ... }
```

Mais cette route est un **facade/helper** qui appelle les routes de base en interne, pas une nouvelle responsabilité.

### Conclusion

**Recommandation finale : Option B (séquence manuelle)**

- ✅ Respect des principes de design
- ✅ Transparence de l'API
- ✅ Flexibilité pour le backend métier
- ✅ Facilite la maintenance et l'évolution

Le backend métier gère la séquence, mais il peut créer des helpers/facades côté client pour simplifier l'utilisation dans son code.

