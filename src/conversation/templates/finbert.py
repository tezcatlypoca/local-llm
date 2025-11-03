"""
Template pour FinBERT.

Note: FinBERT est principalement un modèle de classification pour la finance.
Pour un usage en chat, on utilise un format simple compatible avec les modèles
BERT-based pour génération.
"""
from typing import List, Optional
from .base import Template
from ..context import Message, MessageRole


class FinBertTemplate(Template):
    """
    Template pour FinBERT.
    
    FinBERT étant un modèle basé sur BERT (encoder-only), il n'a pas de format
    de chat natif comme les modèles génératifs. Pour un usage conversationnel,
    on utilise un format simple avec des séparateurs.
    
    Format simple :
    [SYSTEM] {system_prompt}
    [USER] {user_message}
    [ASSISTANT] {assistant_message}
    
    Ou format encore plus simple pour BERT :
    {system_prompt}
    User: {user_message}
    Assistant: {assistant_message}
    """
    
    def get_system_token(self) -> Optional[str]:
        return "[SYSTEM]"
    
    def get_user_token(self) -> str:
        return "[USER]"
    
    def get_assistant_token(self) -> str:
        return "[ASSISTANT]"
    
    def get_end_token(self) -> Optional[str]:
        return None  # Pas de token de fin pour FinBERT
    
    def format_message(self, role: MessageRole, content: str) -> str:
        """Formate un message selon son rôle."""
        if role == MessageRole.SYSTEM:
            return f"{self.system_token} {content}\n"
        elif role == MessageRole.USER:
            return f"{self.user_token} {content}\n"
        elif role == MessageRole.ASSISTANT:
            return f"{self.assistant_token} {content}\n"
        return content
    
    def format_conversation(
        self, 
        messages: List[Message], 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Formate une conversation complète pour FinBERT.
        """
        parts = []
        
        # Ajouter le prompt système s'il existe
        if system_prompt:
            parts.append(f"{self.system_token} {system_prompt}\n")
        else:
            # Chercher un message système dans la liste
            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    parts.append(self.format_message(msg.role, msg.content))
                    break
        
        # Ajouter tous les messages (hors système si déjà ajouté)
        system_added = bool(system_prompt)
        for msg in messages:
            if msg.role == MessageRole.SYSTEM and system_added:
                continue
            parts.append(self.format_message(msg.role, msg.content))
        
        return "".join(parts)

