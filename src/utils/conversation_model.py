from dataclasses import dataclass, field
from typing import List


@dataclass
class ConversationModel:
    id: int
    name: str
    message: List[str] = field(default_factory=list)
    temperature: float = 0.0
    message_max: int = 0