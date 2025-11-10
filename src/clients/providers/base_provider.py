"""
Interface abstraite pour les providers d'API (Local, Groq, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.utils.models.message_model import MessageModel


class BaseProvider(ABC):
    """
    Classe abstraite définissant l'interface commune pour tous les providers d'API.
    
    Un provider est responsable de :
    - L'envoi de messages à l'API
    - La récupération des modèles disponibles
    - La vérification de disponibilité
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Retourne le nom du provider (ex: "local", "groq").
        
        Returns:
            Nom du provider
        """
        pass
    
    @abstractmethod
    def send_message(
        self,
        messages: List[MessageModel],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Envoie un message (ou une liste de messages) à l'API et retourne la réponse.
        
        Args:
            messages: Liste de messages (contexte de la conversation)
            model_name: Nom du modèle à utiliser
            temperature: Température pour la génération (0.0-2.0)
            max_tokens: Nombre maximum de tokens à générer (optionnel)
            **kwargs: Paramètres additionnels spécifiques au provider
        
        Returns:
            Contenu de la réponse de l'assistant (string)
        
        Raises:
            ValueError: Si le modèle n'est pas disponible ou si les paramètres sont invalides
            ConnectionError: Si l'API n'est pas accessible
            Exception: Autres erreurs spécifiques au provider
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des modèles disponibles pour ce provider.
        
        Returns:
            Liste de dictionnaires contenant les informations sur les modèles.
            Format attendu : [{"identifier": "model-name", ...}, ...]
        
        Raises:
            ConnectionError: Si l'API n'est pas accessible
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Vérifie si le provider est disponible et accessible.
        
        Returns:
            True si le provider est disponible, False sinon
        """
        pass
    
    def validate_model(self, model_name: str) -> bool:
        """
        Vérifie si un modèle est disponible pour ce provider.
        
        Args:
            model_name: Nom du modèle à vérifier
        
        Returns:
            True si le modèle est disponible, False sinon
        """
        try:
            available_models = self.get_available_models()
            model_identifiers = [model.get("identifier", "") for model in available_models]
            return model_name in model_identifiers
        except Exception:
            return False
    
    def _messages_to_dict_list(self, messages: List[MessageModel]) -> List[Dict[str, str]]:
        """
        Convertit une liste de MessageModel en liste de dictionnaires.
        Méthode utilitaire pour les providers qui utilisent le format messages array.
        
        Args:
            messages: Liste de MessageModel
        
        Returns:
            Liste de dictionnaires au format [{"role": "user", "content": "..."}, ...]
        """
        return MessageModel.messages_to_dict(messages)

