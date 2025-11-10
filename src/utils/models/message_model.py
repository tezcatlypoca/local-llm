from dataclasses import dataclass 
from typing import Dict, List, Any
from enum import Enum

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class MessageModel:
    id: int
    conversation_id: int
    role: Role
    content: str
    temperature: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le modèle en dictionnaire pour la sérialisation JSON.
        
        Returns:
            Dictionnaire représentant le message
        """
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role.value,
            "content": self.content,
            "temperature": self.temperature
        }

    @classmethod
    def message_to_dict(cls, message) -> Dict[str, str]:
        return {
            "role": message.role.value,
            "content": message.content
        }
    
    @classmethod
    def messages_to_dict(cls, messages: List) -> List[Dict[str, str]]:
        messages_dict = []
        for message in messages:
            messages_dict.append({
                "role": message.role.value,
                "content": message.content
            })
        return messages_dict