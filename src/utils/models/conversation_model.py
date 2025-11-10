from src.utils.models.message_model import MessageModel
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ConversationModel:
    id: int
    name: str
    model_name: str
    message: List[MessageModel] = field(default_factory=list)
    provider: str = "local"  # Provider à utiliser : "local" ou "groq"
    temperature: float = 0.7
    message_max: int = 0
    system_prompt: Optional[str] = field(default=None)  # Message système pour guider le modèle
    
    # Cache du formatage pour optimiser les performances
    _formatted_cache: Optional[str] = field(default=None, repr=False)
    _cache_version: int = field(default=0, repr=False)  # Nombre de messages au moment du cache
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le modèle en dictionnaire pour la sérialisation JSON.
        Exclut les champs de cache internes.
        
        Returns:
            Dictionnaire représentant la conversation
        """
        return {
            'id': self.id,
            'name': self.name,
            'model_name': self.model_name,
            'provider': self.provider,
            'temperature': self.temperature,
            'message_max': self.message_max,
            'system_prompt': self.system_prompt,
            'messages': [msg.to_dict() for msg in self.message]
        }