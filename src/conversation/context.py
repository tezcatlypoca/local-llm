"""
Gestion du contexte de conversation.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(Enum):
    """Rôles des messages dans une conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Représente un message dans une conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le message pour la persistance."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Désérialise un message depuis la persistance."""
        if isinstance(data["timestamp"], str):
            timestamp = datetime.fromisoformat(data["timestamp"])
        else:
            timestamp = data["timestamp"]
        
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


class ConversationContext:
    """
    Gère le contexte d'une conversation.
    
    Responsabilités :
    - Stocker l'historique des messages
    - Gérer la limite de messages (truncation simple)
    - Préparer le contexte pour le formatage
    
    Note : Conçu pour permettre l'évolution vers une gestion par résumé.
    """
    
    def __init__(
        self,
        conversation_id: str,
        model_name: str,
        gpu_id: int,
        max_messages: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialise un contexte de conversation.
        
        Args:
            conversation_id: Identifiant unique de la conversation
            model_name: Nom du modèle utilisé
            gpu_id: ID du GPU utilisé
            max_messages: Nombre maximum de messages à garder (None = pas de limite)
            system_prompt: Prompt système optionnel
        """
        self.conversation_id = conversation_id
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.system_prompt = system_prompt
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        # Champ pour stocker un résumé (pour évolution future)
        self.summary: Optional[str] = None
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Ajoute un message au contexte.
        
        Applique automatiquement la troncature si nécessaire.
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        # Appliquer la troncature si nécessaire
        self._truncate_if_needed()
    
    def get_messages(self, include_system: bool = False) -> List[Message]:
        """
        Retourne tous les messages.
        
        Args:
            include_system: Si True, inclut les messages système dans la liste
        
        Returns:
            Liste des messages (hors système par défaut)
        """
        if include_system:
            return self.messages.copy()
        return [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]
    
    def get_messages_for_formatting(self) -> List[Message]:
        """
        Retourne les messages prêts pour le formatage.
        
        Cette méthode peut être surchargée dans une version future pour
        inclure un résumé à la place des anciens messages.
        
        Returns:
            Liste des messages à formater (incluant le système si présent)
        """
        # Version simple : retourne tous les messages
        # Version résumé (future) : retournerait [summary_message] + messages_récents
        messages = []
        
        # Si on a un résumé (future évolution), on l'ajouterait ici
        if self.summary:
            # Pour l'instant on ignore le résumé en mode simple
            # Dans la version résumé, on créerait un message système avec le résumé
            pass
        
        return self.messages.copy()
    
    def _truncate_if_needed(self):
        """
        Tronque le contexte si le nombre de messages dépasse max_messages.
        
        Stratégie simple : garde les N derniers messages.
        Conserve toujours le message système s'il existe.
        
        Note : Cette méthode peut être remplacée par une version qui génère
        un résumé des anciens messages au lieu de les supprimer.
        """
        if self.max_messages is None:
            return
        
        # Séparer les messages système des autres
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        other_messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]
        
        # Garder les N derniers messages non-système
        if len(other_messages) > self.max_messages:
            # Version simple : on garde juste les N derniers
            truncated = other_messages[-self.max_messages:]
            
            # Version résumé (future) : ici on pourrait :
            # 1. Créer un résumé des messages supprimés
            # 2. Stocker ce résumé dans self.summary
            # 3. Garder tous les messages récents
            
            self.messages = system_messages + truncated
        
        # Si on a un système prompt mais pas de message système, on pourrait
        # vouloir le convertir en message (selon le template utilisé)
    
    def truncate_context(self, target_count: int):
        """
        Force la troncature du contexte à un nombre de messages donné.
        
        Args:
            target_count: Nombre de messages à garder (hors système)
        """
        self.max_messages = target_count
        self._truncate_if_needed()
    
    def clear_context(self, keep_system: bool = True):
        """
        Vide le contexte.
        
        Args:
            keep_system: Si True, conserve les messages système
        """
        if keep_system:
            self.messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        else:
            self.messages = []
        
        self.summary = None
        self.updated_at = datetime.now()
    
    def get_message_count(self) -> int:
        """Retourne le nombre de messages (hors système)."""
        return len([msg for msg in self.messages if msg.role != MessageRole.SYSTEM])
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le contexte pour la persistance."""
        return {
            "conversation_id": self.conversation_id,
            "model_name": self.model_name,
            "gpu_id": self.gpu_id,
            "max_messages": self.max_messages,
            "system_prompt": self.system_prompt,
            "messages": [msg.to_dict() for msg in self.messages],
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Désérialise un contexte depuis la persistance."""
        context = cls(
            conversation_id=data["conversation_id"],
            model_name=data["model_name"],
            gpu_id=data["gpu_id"],
            max_messages=data.get("max_messages"),
            system_prompt=data.get("system_prompt")
        )
        
        context.messages = [
            Message.from_dict(msg_data) 
            for msg_data in data.get("messages", [])
        ]
        context.summary = data.get("summary")
        
        if isinstance(data.get("created_at"), str):
            context.created_at = datetime.fromisoformat(data["created_at"])
        else:
            context.created_at = data.get("created_at", datetime.now())
        
        if isinstance(data.get("updated_at"), str):
            context.updated_at = datetime.fromisoformat(data["updated_at"])
        else:
            context.updated_at = data.get("updated_at", datetime.now())
        
        return context

