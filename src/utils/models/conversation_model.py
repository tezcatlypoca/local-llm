from src.utils.models.message_model import MessageModel
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConversationModel:
    id: int
    name: str
    message: List[MessageModel] = field(default_factory=list)
    model_name: str
    temperature: float = 0.7
    message_max: int = 0
    
    # Cache du formatage pour optimiser les performances
    _formatted_cache: Optional[str] = field(default=None, repr=False)
    _cache_version: int = field(default=0, repr=False)  # Nombre de messages au moment du cache