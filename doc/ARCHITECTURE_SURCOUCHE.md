# Architecture de la Surcouche API - Gestion Contexte, Templates et Persistance

## Vue d'ensemble

Cette surcouche s'interface avec `LLMClient` pour gérer :
1. **Contexte de conversation** : Historique des échanges utilisateur/assistant
2. **Templates adaptés** : Formatage spécifique pour chaque modèle (GPT-2, Phi-2, TinyLlama, etc.)
3. **Persistance** : Sauvegarde des conversations et métadonnées

---

## Architecture proposée

### Structure des modules

```
src/
├── client/                    # Client API de base (existant)
│   └── ...
└── conversation/              # Nouveau module - Surcouche
    ├── __init__.py
    ├── manager.py             # Gestionnaire principal de conversations
    ├── context.py             # Gestion du contexte/historique
    ├── templates/             # Templates pour chaque modèle
    │   ├── __init__.py
    │   ├── base.py            # Classe abstraite pour templates
    │   ├── gpt2.py            # Template GPT-2
    │   ├── phi2.py            # Template Phi-2 (Microsoft)
    │   ├── tinyllama.py       # Template TinyLlama
    │   ├── mistral.py         # Template Mistral
    │   └── registry.py        # Registre des templates par modèle
    ├── storage/               # Persistance des données
    │   ├── __init__.py
    │   ├── base.py            # Interface abstraite de stockage
    │   ├── json_storage.py    # Stockage JSON (fichiers)
    │   ├── sqlite_storage.py  # Stockage SQLite (recommandé)
    │   └── models.py          # Modèles de données (Conversation, Message, etc.)
    └── exceptions.py          # Exceptions personnalisées
```

---

## 1. Gestion du Contexte

### Concept

Le **contexte** représente l'historique d'une conversation avec un modèle. Il doit :
- Stocker les messages dans l'ordre chronologique
- Gérer les rôles (system, user, assistant)
- Limiter la taille du contexte (troncature intelligente)
- Permettre l'ajout/suppression de messages

### Implémentation proposée

#### Classe `ConversationContext`

```python
# src/conversation/context.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Literal
from enum import Enum

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    """Représente un message dans une conversation."""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Sérialise le message pour la persistance."""
        pass
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Désérialise un message depuis la persistance."""
        pass

class ConversationContext:
    """
    Gère le contexte d'une conversation.
    
    Responsabilités :
    - Stocker l'historique des messages
    - Gérer la limite de tokens/context window
    - Formater le contexte selon le template du modèle
    """
    
    def __init__(
        self,
        conversation_id: str,
        model_name: str,
        max_context_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        self.conversation_id = conversation_id
        self.model_name = model_name
        self.messages: List[Message] = []
        self.max_context_tokens = max_context_tokens
        self.system_prompt = system_prompt
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None):
        """Ajoute un message au contexte."""
        pass
    
    def get_messages(self) -> List[Message]:
        """Retourne tous les messages."""
        pass
    
    def truncate_context(self, target_tokens: int):
        """
        Tronque le contexte pour respecter la limite de tokens.
        
        Stratégie :
        - Conserver le message système (si présent)
        - Conserver les messages récents
        - Optionnel : Résumer les anciens messages
        """
        pass
    
    def clear_context(self):
        """Vide le contexte (garde le prompt système si présent)."""
        pass
    
    def to_formatted_string(self, template) -> str:
        """
        Formate le contexte selon le template du modèle.
        
        Args:
            template: Instance du template spécifique au modèle
        """
        pass
```

### Gestion de la taille du contexte

**Stratégies de troncature :**

1. **Troncature simple** : Garde les N derniers messages
2. **Troncature par tokens** : Estime le nombre de tokens et tronque
3. **Résumé progressif** : Résume les anciens messages (avancé)
4. **Fenêtre glissante** : Garde un certain nombre de messages récents + anciens importants

**Recommandation** : Implémenter d'abord la troncature simple, puis ajouter la gestion par tokens si nécessaire.

---

## 2. Système de Templates

### Concept

Chaque modèle LLM attend un format de prompt spécifique :
- **GPT-2** : Format simple sans séparateurs spéciaux
- **Phi-2** : Format `<|user|>`, `<|assistant|>`, `<|end|>`
- **TinyLlama** : Format ChatML `<|im_start|>user`, `<|im_end|>`
- **Mistral** : Format avec `[INST]` et `[/INST]`

### Implémentation proposée

#### Classe abstraite `Template`

```python
# src/conversation/templates/base.py

from abc import ABC, abstractmethod
from typing import List
from ..context import Message, MessageRole

class Template(ABC):
    """Classe abstraite pour les templates de formatage."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.system_token = self.get_system_token()
        self.user_token = self.get_user_token()
        self.assistant_token = self.get_assistant_token()
        self.end_token = self.get_end_token()
    
    @abstractmethod
    def get_system_token(self) -> Optional[str]:
        """Retourne le token pour les messages système."""
        pass
    
    @abstractmethod
    def get_user_token(self) -> str:
        """Retourne le token pour les messages utilisateur."""
        pass
    
    @abstractmethod
    def get_assistant_token(self) -> str:
        """Retourne le token pour les messages assistant."""
        pass
    
    @abstractmethod
    def get_end_token(self) -> Optional[str]:
        """Retourne le token de fin (si applicable)."""
        pass
    
    @abstractmethod
    def format_message(self, role: MessageRole, content: str) -> str:
        """Formate un message unique selon le rôle."""
        pass
    
    @abstractmethod
    def format_conversation(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """
        Formate une conversation complète.
        
        Args:
            messages: Liste des messages à formater
            system_prompt: Prompt système optionnel
        
        Returns:
            String formatée prête à être envoyée au modèle
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estime le nombre de tokens dans un texte.
        
        Approximation : 1 token ≈ 4 caractères pour l'anglais
        Pour plus de précision, utiliser un tokenizer si disponible.
        """
        return len(text) // 4
```

#### Exemple : Template GPT-2

```python
# src/conversation/templates/gpt2.py

from typing import Optional
from .base import Template
from ..context import Message, MessageRole

class GPT2Template(Template):
    """
    Template pour GPT-2.
    
    Format simple : pas de tokens spéciaux, juste le texte.
    Les messages sont concaténés avec des retours à la ligne.
    """
    
    def get_system_token(self) -> Optional[str]:
        return None
    
    def get_user_token(self) -> str:
        return ""  # Pas de token pour GPT-2
    
    def get_assistant_token(self) -> str:
        return ""
    
    def get_end_token(self) -> Optional[str]:
        return None
    
    def format_message(self, role: MessageRole, content: str) -> str:
        """GPT-2 n'a pas de formatage spécial."""
        return content
    
    def format_conversation(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        parts = []
        
        if system_prompt:
            parts.append(system_prompt)
        
        for msg in messages:
            parts.append(msg.content)
        
        return "\n".join(parts)
```

#### Exemple : Template Phi-2

```python
# src/conversation/templates/phi2.py

from typing import Optional
from .base import Template
from ..context import Message, MessageRole

class Phi2Template(Template):
    """
    Template pour Phi-2 (Microsoft).
    
    Format :
    <|system|>
    {system_prompt}
    <|user|>
    {user_message}
    <|assistant|>
    {assistant_message}
    <|end|>
    """
    
    def get_system_token(self) -> Optional[str]:
        return "<|system|>"
    
    def get_user_token(self) -> str:
        return "<|user|>"
    
    def get_assistant_token(self) -> str:
        return "<|assistant|>"
    
    def get_end_token(self) -> Optional[str]:
        return "<|end|>"
    
    def format_message(self, role: MessageRole, content: str) -> str:
        if role == MessageRole.SYSTEM:
            return f"{self.system_token}\n{content}"
        elif role == MessageRole.USER:
            return f"{self.user_token}\n{content}"
        elif role == MessageRole.ASSISTANT:
            return f"{self.assistant_token}\n{content}"
        return content
    
    def format_conversation(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        parts = []
        
        if system_prompt:
            parts.append(f"{self.system_token}\n{system_prompt}")
        
        for msg in messages:
            parts.append(self.format_message(msg.role, msg.content))
        
        if self.end_token:
            parts.append(self.end_token)
        
        return "\n".join(parts)
```

#### Registry des Templates

```python
# src/conversation/templates/registry.py

from typing import Dict, Type, Optional
from .base import Template
from .gpt2 import GPT2Template
from .phi2 import Phi2Template
from .tinyllama import TinyLlamaTemplate

class TemplateRegistry:
    """
    Registre centralisé des templates.
    
    Permet de récupérer automatiquement le bon template
    en fonction du nom du modèle.
    """
    
    # Mapping nom modèle -> classe template
    _template_classes: Dict[str, Type[Template]] = {
        "gpt2": GPT2Template,
        "microsoft/phi-2": Phi2Template,
        "microsoft/phi-1.5": Phi2Template,  # Même format
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": TinyLlamaTemplate,
        # Ajouter d'autres modèles...
    }
    
    # Patterns pour détecter le type de modèle
    _model_patterns: Dict[str, Type[Template]] = {
        "gpt2": GPT2Template,
        "phi": Phi2Template,
        "tinyllama": TinyLlamaTemplate,
        "mistral": MistralTemplate,
    }
    
    @classmethod
    def get_template(cls, model_name: str) -> Template:
        """
        Récupère le template approprié pour un modèle.
        
        Args:
            model_name: Nom du modèle (ex: "gpt2", "microsoft/phi-2")
        
        Returns:
            Instance du template approprié
        
        Raises:
            TemplateNotFoundError: Si aucun template n'est trouvé
        """
        # Recherche exacte
        if model_name in cls._template_classes:
            template_class = cls._template_classes[model_name]
            return template_class(model_name)
        
        # Recherche par pattern
        model_lower = model_name.lower()
        for pattern, template_class in cls._model_patterns.items():
            if pattern in model_lower:
                return template_class(model_name)
        
        # Fallback : GPT-2 (format simple)
        return GPT2Template(model_name)
    
    @classmethod
    def register_template(cls, model_name: str, template_class: Type[Template]):
        """Enregistre un nouveau template personnalisé."""
        cls._template_classes[model_name] = template_class
```

---

## 3. Persistance des Échanges

### Concept

La persistance permet de :
- Sauvegarder les conversations complètes
- Reprendre une conversation interrompue
- Analyser l'historique d'utilisation
- Exporter les données

### Stratégies de stockage

#### Option 1 : SQLite (Recommandé)

**Avantages :**
- Base de données relationnelle légère
- Pas de serveur requis
- Requêtes SQL pour recherches/filtres
- Gestion ACID
- Scalable jusqu'à plusieurs GB

**Structure proposée :**

```sql
-- Table des conversations
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    gpu_id INTEGER,
    system_prompt TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata TEXT  -- JSON pour données supplémentaires
);

-- Table des messages
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'system', 'user', 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP,
    metadata TEXT,  -- JSON
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Index pour améliorer les performances
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
```

#### Option 2 : JSON (Fichiers)

**Avantages :**
- Simple à implémenter
- Lisible par l'humain
- Pas de dépendance externe

**Inconvénients :**
- Peu performant pour grandes quantités
- Pas de requêtes complexes
- Risque de corruption si écriture interrompue

**Structure proposée :**

```json
{
  "conversations": [
    {
      "id": "conv_123",
      "model_name": "microsoft/phi-2",
      "gpu_id": 0,
      "created_at": "2024-01-15T10:00:00",
      "messages": [
        {
          "role": "user",
          "content": "Bonjour",
          "timestamp": "2024-01-15T10:00:05"
        }
      ]
    }
  ]
}
```

### Implémentation proposée

#### Interface abstraite

```python
# src/conversation/storage/base.py

from abc import ABC, abstractmethod
from typing import List, Optional
from ..context import ConversationContext, Message

class StorageBackend(ABC):
    """Interface abstraite pour le stockage."""
    
    @abstractmethod
    def save_conversation(self, context: ConversationContext):
        """Sauvegarde une conversation."""
        pass
    
    @abstractmethod
    def load_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Charge une conversation par son ID."""
        pass
    
    @abstractmethod
    def list_conversations(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """Liste les IDs des conversations (optionnellement filtrées)."""
        pass
    
    @abstractmethod
    def delete_conversation(self, conversation_id: str):
        """Supprime une conversation."""
        pass
    
    @abstractmethod
    def add_message(self, conversation_id: str, message: Message):
        """Ajoute un message à une conversation existante."""
        pass
```

#### Implémentation SQLite

```python
# src/conversation/storage/sqlite_storage.py

import sqlite3
import json
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from .base import StorageBackend
from ..context import ConversationContext, Message, MessageRole

class SQLiteStorage(StorageBackend):
    """
    Stockage SQLite pour les conversations.
    
    Crée automatiquement la base de données et les tables
    si elles n'existent pas.
    """
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Crée les tables si elles n'existent pas."""
        # Implémentation...
        pass
    
    def save_conversation(self, context: ConversationContext):
        """Sauvegarde une conversation complète."""
        # Implémentation...
        pass
    
    # ... autres méthodes
```

---

## 4. Gestionnaire Principal

### Classe `ConversationManager`

```python
# src/conversation/manager.py

from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime

from ..client import LLMClient
from .context import ConversationContext, Message, MessageRole
from .templates.registry import TemplateRegistry
from .storage.base import StorageBackend
from .storage.sqlite_storage import SQLiteStorage

class ConversationManager:
    """
    Gestionnaire principal de conversations.
    
    Point d'entrée pour :
    - Créer/gérer des conversations
    - Envoyer des messages avec formatage automatique
    - Gérer le contexte
    - Persister les échanges
    """
    
    def __init__(
        self,
        client: LLMClient,
        storage: Optional[StorageBackend] = None,
        auto_save: bool = True
    ):
        self.client = client
        self.storage = storage or SQLiteStorage()
        self.auto_save = auto_save
        self._active_conversations: Dict[str, ConversationContext] = {}
    
    def create_conversation(
        self,
        model_name: str,
        gpu_id: int,
        system_prompt: Optional[str] = None,
        max_context_tokens: Optional[int] = None
    ) -> str:
        """
        Crée une nouvelle conversation.
        
        Returns:
            conversation_id: ID unique de la conversation
        """
        conversation_id = str(uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            system_prompt=system_prompt
        )
        
        self._active_conversations[conversation_id] = context
        
        if self.auto_save:
            self.storage.save_conversation(context)
        
        return conversation_id
    
    def send_message(
        self,
        conversation_id: str,
        user_message: str,
        temperature: float = 0.7,
        max_new_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Envoie un message et récupère la réponse.
        
        Processus :
        1. Récupère le contexte de la conversation
        2. Ajoute le message utilisateur au contexte
        3. Formate le contexte selon le template du modèle
        4. Envoie à l'API via LLMClient
        5. Ajoute la réponse au contexte
        6. Sauvegarde (si auto_save activé)
        
        Returns:
            {
                "response": str,
                "conversation_id": str,
                "gpu_id": int,
                ...
            }
        """
        # 1. Récupérer le contexte
        context = self.get_conversation(conversation_id)
        if not context:
            raise ConversationNotFoundError(conversation_id)
        
        # 2. Ajouter message utilisateur
        context.add_message(MessageRole.USER, user_message)
        
        # 3. Obtenir le template
        template = TemplateRegistry.get_template(context.model_name)
        
        # 4. Formater le contexte
        formatted_prompt = template.format_conversation(
            context.get_messages(),
            context.system_prompt
        )
        
        # 5. Envoyer à l'API
        # TODO: Récupérer gpu_id depuis le contexte ou en paramètre
        response_data = self.client.chat.send_message(
            gpu_id=0,  # À récupérer depuis le contexte
            message=formatted_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        # 6. Extraire la réponse
        assistant_response = response_data["response"]
        
        # 7. Ajouter au contexte
        context.add_message(MessageRole.ASSISTANT, assistant_response)
        
        # 8. Sauvegarder
        if self.auto_save:
            self.storage.save_conversation(context)
        
        return {
            "response": assistant_response,
            "conversation_id": conversation_id,
            "model_name": context.model_name,
            **response_data
        }
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Récupère une conversation (active ou depuis le stockage)."""
        # Vérifier d'abord les conversations actives
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]
        
        # Charger depuis le stockage
        context = self.storage.load_conversation(conversation_id)
        if context:
            self._active_conversations[conversation_id] = context
        
        return context
    
    def list_conversations(self, model_name: Optional[str] = None) -> List[str]:
        """Liste les IDs des conversations."""
        return self.storage.list_conversations(model_name=model_name)
```

---

## 5. Exemple d'utilisation

```python
from src.client import LLMClient
from src.conversation import ConversationManager

# Initialiser le client API
client = LLMClient(base_url="http://192.168.1.100:5000")

# Initialiser le gestionnaire de conversations
manager = ConversationManager(client, auto_save=True)

# Charger un modèle sur un GPU
result = client.models.load_model("microsoft/phi-2")
gpu_id = result["gpu_id"]

# Créer une nouvelle conversation
conv_id = manager.create_conversation(
    model_name="microsoft/phi-2",
    gpu_id=gpu_id,
    system_prompt="Tu es un assistant IA utile et bienveillant."
)

# Envoyer des messages
response1 = manager.send_message(
    conversation_id=conv_id,
    user_message="Bonjour !"
)
print(response1["response"])

response2 = manager.send_message(
    conversation_id=conv_id,
    user_message="Qu'est-ce que l'intelligence artificielle ?"
)
print(response2["response"])

# Reprendre une conversation existante
old_conv = manager.get_conversation(conv_id)
# Le contexte est automatiquement restauré
```

---

## 6. Suggestions et Améliorations Futures

### A. Gestion avancée du contexte

1. **Estimation précise des tokens** : Intégrer un tokenizer réel au lieu de l'approximation
2. **Résumé automatique** : Utiliser un modèle léger pour résumer les anciens messages
3. **Système de priorités** : Marquer certains messages comme "importants" (à ne pas tronquer)

### B. Métadonnées enrichies

```python
@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Nouveaux champs :
    token_count: Optional[int] = None
    generation_time: Optional[float] = None
    temperature_used: Optional[float] = None
    model_version: Optional[str] = None
```

### C. Export/Import

- Export en JSON/Markdown/CSV
- Import de conversations depuis d'autres formats
- Synchronisation avec d'autres systèmes

### D. Recherche et analyse

- Recherche full-text dans les conversations
- Statistiques d'utilisation (tokens, temps, etc.)
- Analytics par modèle

### E. Multi-utilisateurs (optionnel)

- Séparation des conversations par utilisateur
- Authentification/autorisation
- Quotas d'utilisation

### F. Cache et performance

- Cache en mémoire des conversations actives
- Lazy loading des messages
- Pagination pour grandes conversations

### G. Validation et erreurs

```python
# src/conversation/exceptions.py

class ConversationError(Exception):
    """Exception de base pour les erreurs de conversation."""
    pass

class ConversationNotFoundError(ConversationError):
    """Conversation introuvable."""
    pass

class TemplateNotFoundError(ConversationError):
    """Template non trouvé pour le modèle."""
    pass

class ContextTooLargeError(ConversationError):
    """Le contexte dépasse la limite autorisée."""
    pass
```

---

## 7. Configuration

### Fichier de configuration

```python
# src/conversation/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ConversationConfig:
    """Configuration pour le système de conversations."""
    
    # Stockage
    storage_type: str = "sqlite"  # "sqlite" ou "json"
    storage_path: str = "conversations.db"
    
    # Contexte
    default_max_context_tokens: Optional[int] = None
    truncation_strategy: str = "simple"  # "simple", "token_based", "summarize"
    
    # Templates
    template_registry_path: Optional[str] = None  # Pour templates personnalisés
    
    # Performance
    cache_size: int = 100  # Nombre de conversations en cache
    
    # Auto-sauvegarde
    auto_save: bool = True
    auto_save_interval: int = 5  # Sauvegarde toutes les N secondes
```

---

## 8. Tests

### Structure de tests proposée

```
src/tests/
├── test_conversation/
│   ├── __init__.py
│   ├── test_context.py
│   ├── test_templates.py
│   ├── test_storage.py
│   └── test_manager.py
```

### Exemples de tests

- Test de formatage de templates
- Test de troncature de contexte
- Test de persistance SQLite/JSON
- Test d'intégration end-to-end

---

## 9. Documentation

- Docstrings complètes pour toutes les classes/méthodes
- Guide d'utilisation avec exemples
- Documentation des templates supportés
- Guide de migration (si changement de format)

---

## Résumé des fichiers à créer

1. **Core :**
   - `src/conversation/manager.py` - Gestionnaire principal
   - `src/conversation/context.py` - Gestion du contexte
   - `src/conversation/exceptions.py` - Exceptions

2. **Templates :**
   - `src/conversation/templates/base.py` - Classe abstraite
   - `src/conversation/templates/gpt2.py`
   - `src/conversation/templates/phi2.py`
   - `src/conversation/templates/tinyllama.py`
   - `src/conversation/templates/registry.py`

3. **Stockage :**
   - `src/conversation/storage/base.py` - Interface abstraite
   - `src/conversation/storage/sqlite_storage.py`
   - `src/conversation/storage/json_storage.py`
   - `src/conversation/storage/models.py` - Modèles de données

4. **Configuration :**
   - `src/conversation/config.py`

---

## Ordre d'implémentation recommandé

1. **Phase 1 : Fondations**
   - Classes de base (`Message`, `ConversationContext`)
   - Template simple (GPT-2) et registry
   - Stockage JSON (le plus simple)

2. **Phase 2 : Templates**
   - Templates pour Phi-2, TinyLlama
   - Tests de formatage

3. **Phase 3 : Gestionnaire**
   - `ConversationManager` avec intégration API
   - Tests d'intégration

4. **Phase 4 : Persistance avancée**
   - Migration vers SQLite
   - Requêtes et recherche

5. **Phase 5 : Améliorations**
   - Gestion avancée du contexte
   - Métadonnées enrichies
   - Export/Import

---

Cette architecture est modulaire, extensible et permet d'ajouter facilement de nouveaux modèles ou fonctionnalités.

