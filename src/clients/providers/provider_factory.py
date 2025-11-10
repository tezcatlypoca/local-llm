"""
Factory pour créer et gérer les instances de providers.
"""

from typing import Dict, Optional
from src.clients.providers.base_provider import BaseProvider
from src.clients.providers.local_provider import LocalProvider
from src.clients.providers.groq_provider import GroqProvider
import logging
import os

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory pour créer les instances de providers.
    
    Les providers sont des objets légers et peuvent être réutilisés.
    Cette factory gère un cache des instances pour éviter de les recréer.
    """
    
    # Cache des instances de providers (singleton pattern)
    _providers: Dict[str, BaseProvider] = {}
    
    @classmethod
    def get_provider(cls, provider_name: str) -> BaseProvider:
        """
        Récupère ou crée un provider selon son nom.
        
        Les providers sont mis en cache et réutilisés (singleton pattern).
        Pas besoin de les détruire, ce sont des objets légers.
        
        Args:
            provider_name: Nom du provider ("local", "groq", etc.)
        
        Returns:
            Instance du provider demandé
        
        Raises:
            ValueError: Si le provider n'existe pas
        """
        # Normaliser le nom (minuscules)
        provider_name = provider_name.lower()
        
        # Vérifier le cache
        if provider_name in cls._providers:
            return cls._providers[provider_name]
        
        # Créer le provider selon le nom
        if provider_name == "local":
            provider = LocalProvider()
        elif provider_name == "groq":
            provider = GroqProvider()
        else:
            raise ValueError(
                f"Provider '{provider_name}' non reconnu. "
                f"Providers disponibles: {cls.get_available_providers()}"
            )
        
        # Mettre en cache
        cls._providers[provider_name] = provider
        logger.debug(f"Provider '{provider_name}' créé et mis en cache")
        
        return provider
    
    @classmethod
    def get_default_provider(cls) -> BaseProvider:
        """
        Récupère le provider par défaut.
        
        Le provider par défaut est déterminé par la variable d'environnement
        DEFAULT_API_PROVIDER, ou "local" si non définie.
        
        Returns:
            Instance du provider par défaut
        """
        default_name = os.getenv('DEFAULT_API_PROVIDER', 'local').lower()
        return cls.get_provider(default_name)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Retourne la liste des providers disponibles.
        
        Returns:
            Liste des noms de providers disponibles
        """
        return ["local", "groq"]
    
    @classmethod
    def clear_cache(cls):
        """
        Vide le cache des providers.
        
        Utile pour les tests ou si on veut forcer la recréation.
        """
        cls._providers.clear()
        logger.debug("Cache des providers vidé")

