"""
Classe abstraite de base pour les templates de formatage.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
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
    def format_conversation(
        self, 
        messages: List[Message], 
        system_prompt: Optional[str] = None
    ) -> str:
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

