"""
Interface abstraite pour le stockage des conversations.
"""
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

