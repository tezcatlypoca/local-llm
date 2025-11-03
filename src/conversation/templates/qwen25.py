"""
Template pour Qwen2.5 7B (et variantes quantisées).
"""
from typing import List, Optional
from .base import Template
from ..context import Message, MessageRole


class Qwen25Template(Template):
    """
    Template pour Qwen2.5 7B (quantisé 5-bit et autres variantes).
    
    Format ChatML similaire à TinyLlama mais avec quelques différences :
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    
    Note: Le format est très similaire à TinyLlama, mais conservé séparément
    pour permettre des ajustements spécifiques à Qwen2.5 si nécessaire.
    """
    
    def get_system_token(self) -> Optional[str]:
        return "<|im_start|>system"
    
    def get_user_token(self) -> str:
        return "<|im_start|>user"
    
    def get_assistant_token(self) -> str:
        return "<|im_start|>assistant"
    
    def get_end_token(self) -> Optional[str]:
        return "<|im_end|>"
    
    def format_message(self, role: MessageRole, content: str) -> str:
        """Formate un message selon son rôle."""
        if role == MessageRole.SYSTEM:
            return f"{self.system_token}\n{content}{self.end_token}\n"
        elif role == MessageRole.USER:
            return f"{self.user_token}\n{content}{self.end_token}\n"
        elif role == MessageRole.ASSISTANT:
            return f"{self.assistant_token}\n{content}{self.end_token}\n"
        return content
    
    def format_conversation(
        self, 
        messages: List[Message], 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Formate une conversation complète selon le format ChatML de Qwen2.5.
        """
        parts = []
        
        # Ajouter le prompt système s'il existe
        if system_prompt:
            parts.append(f"{self.system_token}\n{system_prompt}{self.end_token}\n")
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
        
        # Pour la dernière réponse assistant, on ne ferme pas avec <|im_end|>
        # car le modèle va continuer à générer
        result = "".join(parts)
        
        # Retirer le dernier <|im_end|> si c'est une réponse assistant
        if result.endswith(f"{self.end_token}\n") and parts:
            last_part = parts[-1]
            if self.assistant_token in last_part:
                result = result[:-len(f"{self.end_token}\n")]
        
        return result

