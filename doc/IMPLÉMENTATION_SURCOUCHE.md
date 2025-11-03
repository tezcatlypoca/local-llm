# Implémentation de la Surcouche - Documentation

## Vue d'ensemble

La surcouche de gestion de conversations a été implémentée avec les fonctionnalités suivantes :

1. ✅ **Gestion du contexte** : Historique des messages avec troncature simple
2. ✅ **Templates adaptés** : Support pour TinyLlama Chat, FinBERT et Qwen2.5 7B
3. ✅ **Persistance SQLite** : Sauvegarde et reprise des conversations
4. ✅ **Architecture extensible** : Prête pour l'évolution vers le résumé automatique

## Structure des fichiers créés

```
src/conversation/
├── __init__.py                 # Exports publics
├── exceptions.py               # Exceptions personnalisées
├── context.py                  # Gestion du contexte (Message, ConversationContext)
├── manager.py                  # Gestionnaire principal
├── templates/
│   ├── __init__.py
│   ├── base.py                 # Classe abstraite Template
│   ├── tinyllama.py            # Template TinyLlama Chat
│   ├── finbert.py              # Template FinBERT
│   ├── qwen25.py               # Template Qwen2.5
│   └── registry.py             # Registre des templates
└── storage/
    ├── __init__.py
    ├── base.py                 # Interface abstraite
    └── sqlite_storage.py       # Implémentation SQLite
```

## Modèles supportés

### 1. TinyLlama Chat
- **Format** : ChatML avec `<|im_start|>` et `<|im_end|>`
- **Détection** : `tinyllama` dans le nom du modèle

### 2. FinBERT
- **Format** : Simple avec `[SYSTEM]`, `[USER]`, `[ASSISTANT]`
- **Détection** : `finbert` dans le nom du modèle
- **Note** : FinBERT est principalement un modèle de classification, le format est adapté pour un usage conversationnel

### 3. Qwen2.5 7B (quantisé 5-bit et autres)
- **Format** : ChatML similaire à TinyLlama
- **Détection** : `qwen` ou `qwen2.5` dans le nom du modèle

## Utilisation de base

```python
from src.client import LLMClient
from src.conversation import ConversationManager

# Initialiser
client = LLMClient(base_url="http://localhost:5000")
manager = ConversationManager(client, auto_save=True)

# Charger un modèle
result = client.models.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
gpu_id = result["gpu_id"]

# Créer une conversation
conv_id = manager.create_conversation(
    model_name=result["model_name"],
    gpu_id=gpu_id,
    system_prompt="Tu es un assistant utile.",
    max_messages=10  # Garder les 10 derniers messages
)

# Envoyer un message
response = manager.send_message(
    conversation_id=conv_id,
    user_message="Bonjour !"
)
print(response["response"])
```

## Gestion du contexte - Version actuelle

### Stratégie actuelle : Troncature simple

La gestion du contexte utilise une **troncature simple** :
- On garde les N derniers messages (hors système)
- Les messages système sont toujours conservés
- Les anciens messages sont simplement supprimés

**Avantages :**
- Simple et rapide
- Pas de calcul supplémentaire
- Performant

**Limites :**
- Perte d'information pour les conversations longues
- Pas de résumé des anciens échanges

## Évolution future : Résumé automatique

### Analyse de faisabilité ✅

L'évolution vers un système de résumé est **totalement viable** et l'architecture a été conçue pour faciliter cette transition.

### Points d'extension prévus

#### 1. Champ `summary` dans `ConversationContext`

Le champ `summary` existe déjà dans la classe `ConversationContext` :

```python
class ConversationContext:
    # ...
    self.summary: Optional[str] = None  # Prêt pour le résumé
```

#### 2. Méthode `get_messages_for_formatting()`

Cette méthode dans `ConversationContext` peut être étendue :

```python
def get_messages_for_formatting(self) -> List[Message]:
    """Version actuelle : retourne tous les messages"""
    # Version résumé (future) :
    # if self.summary:
    #     summary_msg = Message(MessageRole.SYSTEM, self.summary)
    #     return [summary_msg] + self.messages[-N:]  # Derniers N messages
    return self.messages.copy()
```

#### 3. Méthode `_truncate_if_needed()` extensible

La méthode de troncature peut être remplacée pour générer un résumé :

```python
def _truncate_if_needed(self):
    """Version simple actuelle"""
    if len(other_messages) > self.max_messages:
        # Version résumé (future) :
        # 1. Messages à résumer = autres_messages[:-self.max_messages]
        # 2. Appeler le modèle pour générer le résumé
        # 3. Stocker dans self.summary
        # 4. Garder les messages récents
        truncated = other_messages[-self.max_messages:]
        self.messages = system_messages + truncated
```

### Plan d'implémentation du résumé

#### Étape 1 : Créer une méthode de résumé

```python
class ConversationContext:
    def _generate_summary(self, messages_to_summarize: List[Message], model_manager) -> str:
        """
        Génère un résumé des messages donnés en utilisant le modèle.
        
        Args:
            messages_to_summarize: Messages à résumer
            model_manager: ConversationManager pour appeler le modèle
        
        Returns:
            Résumé textuel
        """
        # Formater les messages à résumer
        summary_prompt = "Résume cette conversation :\n"
        for msg in messages_to_summarize:
            summary_prompt += f"{msg.role.value}: {msg.content}\n"
        
        # Appeler le modèle (peut utiliser un GPU différent ou le même)
        # Note: nécessite un accès au client ou au manager
        response = model_manager._generate_summary(summary_prompt)
        return response
```

#### Étape 2 : Modifier `_truncate_if_needed()`

```python
def _truncate_if_needed(self, enable_summary: bool = False, summary_manager=None):
    """
    Tronque avec option de résumé.
    
    Args:
        enable_summary: Si True, génère un résumé au lieu de supprimer
        summary_manager: Manager pour générer le résumé (peut être le même ou différent)
    """
    if self.max_messages is None:
        return
    
    system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
    other_messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]
    
    if len(other_messages) > self.max_messages:
        if enable_summary and summary_manager:
            # Messages à résumer (les anciens)
            messages_to_summarize = other_messages[:-self.max_messages]
            
            # Générer le résumé
            new_summary = self._generate_summary(messages_to_summarize, summary_manager)
            
            # Combiner avec l'ancien résumé si existe
            if self.summary:
                combined_context = f"{self.summary}\n\n{new_summary}"
            else:
                combined_context = new_summary
            
            self.summary = combined_context
            
            # Garder les messages récents
            recent_messages = other_messages[-self.max_messages:]
            self.messages = system_messages + recent_messages
        else:
            # Version simple (comportement actuel)
            truncated = other_messages[-self.max_messages:]
            self.messages = system_messages + truncated
```

#### Étape 3 : Modifier `get_messages_for_formatting()`

```python
def get_messages_for_formatting(self) -> List[Message]:
    """Retourne les messages à formater, incluant le résumé si disponible."""
    messages = []
    
    # Si on a un résumé, le convertir en message système
    if self.summary:
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"Résumé de la conversation précédente :\n{self.summary}",
            metadata={"is_summary": True}
        )
        messages.append(summary_msg)
    
    # Ajouter tous les messages actuels
    messages.extend(self.messages)
    
    return messages
```

#### Étape 4 : Ajouter un paramètre dans `ConversationManager`

```python
class ConversationManager:
    def __init__(
        self,
        client: LLMClient,
        enable_auto_summary: bool = False,
        summary_model_gpu: Optional[int] = None,  # GPU différent pour le résumé
        ...
    ):
        self.enable_auto_summary = enable_auto_summary
        self.summary_model_gpu = summary_model_gpu
```

### Avantages de cette approche

1. **Rétrocompatibilité** : Le code existant continue de fonctionner
2. **Activation progressive** : Peut être activé par conversation ou globalement
3. **Flexibilité** : Peut utiliser le même modèle ou un modèle dédié pour le résumé
4. **Pas de refonte** : Extension de l'existant, pas de réécriture

### Considérations

1. **Performance** : Générer un résumé prend du temps et des ressources
2. **Qualité** : Dépend de la capacité du modèle à résumer
3. **Coût** : Utilise des tokens supplémentaires
4. **GPU dédié** : Peut nécessiter un deuxième GPU pour le résumé (ou attendre)

### Recommandation

Pour l'implémentation future :
1. Commencer avec un paramètre `enable_summary=False` par défaut
2. Permettre l'activation par conversation
3. Utiliser le même modèle au début (puis optimiser si nécessaire)
4. Ajouter une méthode `generate_summary()` explicite pour contrôler quand générer

## Base de données SQLite

### Schéma

```sql
-- Table conversations
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    gpu_id INTEGER NOT NULL,
    system_prompt TEXT,
    max_messages INTEGER,
    summary TEXT,              -- Pour le résumé futur
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT              -- JSON pour données extensibles
);

-- Table messages
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT,             -- JSON
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```

### Configuration SQLite

- **WAL mode** : Actif pour meilleures performances
- **Foreign keys** : Activées
- **Synchronous** : NORMAL (bon compromis)
- **Indexes** : Sur `conversation_id`, `timestamp`, `model_name`, `updated_at`

### Fichier de base de données

Par défaut : `conversations.db` dans le répertoire courant.

Pour changer :
```python
storage = SQLiteStorage(db_path="chemin/vers/conversations.db")
manager = ConversationManager(client, storage=storage)
```

## Exemples d'utilisation

Voir `doc/exemple_utilisation_conversation.py` pour des exemples complets.

## Prochaines étapes suggérées

1. ✅ **Implémentation de base** - Terminé
2. 🔄 **Tests unitaires** - À créer
3. 🔄 **Gestion du résumé** - À implémenter selon besoin
4. 🔄 **Export/Import** - Pour sauvegarder/conserver les conversations
5. 🔄 **Recherche full-text** - Pour trouver des conversations spécifiques

## Notes importantes

- Le contexte est automatiquement tronqué selon `max_messages`
- Les messages système sont toujours conservés
- Le stockage est automatique si `auto_save=True`
- Les templates sont détectés automatiquement selon le nom du modèle
- L'architecture permet d'ajouter facilement de nouveaux modèles

