"""
Validateurs Pydantic pour les entrées API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CreateConversationRequest(BaseModel):
    """Schéma de validation pour la création d'une conversation."""
    
    # Pas de min_length ici pour permettre au validator personnalisé de gérer les messages d'erreur
    model_name: str = Field(..., max_length=200, description="Nom du modèle à utiliser")
    provider: Optional[str] = Field(default=None, description="Provider à utiliser (local ou groq)")
    name: Optional[str] = Field(default="Conversation", min_length=1, max_length=200, description="Nom de la conversation")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Température pour la génération (0.0-2.0)")
    message_max: Optional[int] = Field(default=10, ge=0, le=1000, description="Nombre maximum de messages")
    system_prompt: Optional[str] = Field(default=None, max_length=2000, description="Message système pour guider le modèle (optionnel, un prompt par défaut sera utilisé si non fourni)")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Valide que le nom du modèle n'est pas vide."""
        if not v or not v.strip():
            raise ValueError("model_name ne peut pas être vide")
        if len(v.strip()) > 200:
            raise ValueError("model_name ne peut pas dépasser 200 caractères")
        return v.strip()
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: Optional[str]) -> Optional[str]:
        """Valide le provider."""
        if v is not None:
            v = v.lower()
            if v not in ['local', 'groq']:
                raise ValueError("provider doit être 'local' ou 'groq'")
        return v


class PostMessageRequest(BaseModel):
    """Schéma de validation pour l'envoi d'un message."""
    
    # Pas de min_length ici pour permettre au validator personnalisé de gérer les messages d'erreur
    content: str = Field(..., max_length=100000, description="Contenu du message")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Valide que le contenu n'est pas vide."""
        # Valider d'abord la longueur avant trim (pour respecter max_length de Field)
        if len(v) > 100000:
            raise ValueError("content ne peut pas dépasser 100000 caractères")
        # Ensuite valider et trim
        v_trimmed = v.strip()
        if not v_trimmed:
            raise ValueError("content ne peut pas être vide")
        return v_trimmed

