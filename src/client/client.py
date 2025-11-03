"""
Client principal pour l'API Local LLM.
"""
import logging
import requests
from typing import Optional
from .config import APIConfig
from .endpoints.root import RootEndpoint
from .endpoints.models import ModelsEndpoint
from .endpoints.health import HealthEndpoint
from .endpoints.chat import ChatEndpoint
from .endpoints.completion import CompletionEndpoint
from .endpoints.logs import LogsEndpoint

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client Python pour interagir avec l'API Local LLM.
    
    Point d'entrée principal pour toutes les interactions avec l'API.
    Fournit un accès organisé à tous les endpoints via des attributs.
    
    Example:
        ```python
        from src.client import LLMClient
        
        # Initialisation avec URL par défaut (http://localhost:5000)
        client = LLMClient()
        
        # Ou avec une URL personnalisée
        client = LLMClient(base_url="http://192.168.1.100:5000")
        
        # Vérifier que l'API est active
        status = client.root.check_status()
        
        # Lister les modèles
        models = client.models.list_models()
        
        # Charger un modèle
        result = client.models.load_model("gpt2")
        access_token = result["access_token"]
        
        # Envoyer un message
        response = client.chat.send_message(
            gpu_id=0,
            message="Bonjour!"
        )
        
        # Décharger le modèle
        client.models.unload_model(gpu_id=0, access_token=access_token)
        ```
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        headers: Optional[dict] = None,
        config: Optional[APIConfig] = None
    ):
        """
        Initialise le client API.
        
        Args:
            base_url: URL de base de l'API (ex: "http://localhost:5000")
                     Si None, utilise la configuration par défaut ou les variables
                     d'environnement
            timeout: Timeout pour les requêtes HTTP en secondes (défaut: 30)
            headers: Headers HTTP personnalisés à inclure dans toutes les requêtes
            config: Instance de APIConfig (alternative à passer base_url/timeout/headers)
        
        Example:
            ```python
            # Initialisation simple
            client = LLMClient()
            
            # Avec URL personnalisée
            client = LLMClient(base_url="http://192.168.1.100:5000")
            
            # Avec configuration complète
            config = APIConfig(base_url="http://localhost:5000", timeout=60)
            client = LLMClient(config=config)
            ```
        """
        # Configuration
        if config is not None:
            self.config = config
        else:
            self.config = APIConfig(
                base_url=base_url,
                timeout=timeout,
                headers=headers
            )
        
        # Session HTTP réutilisable (pour les connexions persistantes)
        self.session = requests.Session()
        if self.config.headers:
            self.session.headers.update(self.config.headers)
        
        # Initialisation des endpoints
        self.root = RootEndpoint(self)
        self.models = ModelsEndpoint(self)
        self.health = HealthEndpoint(self)
        self.chat = ChatEndpoint(self)
        self.completion = CompletionEndpoint(self)
        self.logs = LogsEndpoint(self)
        
        logger.info(
            f"LLMClient initialisé avec base_url='{self.base_url}', timeout={self.timeout}"
        )
    
    @property
    def base_url(self) -> str:
        """Retourne l'URL de base de l'API."""
        return self.config.base_url
    
    @property
    def timeout(self) -> int:
        """Retourne le timeout pour les requêtes HTTP."""
        return self.config.timeout
    
    def close(self):
        """
        Ferme la session HTTP.
        
        À appeler quand vous avez fini d'utiliser le client pour libérer
        les ressources réseau.
        """
        self.session.close()
        logger.debug("Session HTTP fermée")
    
    def __enter__(self):
        """Support du context manager (with statement)."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ferme automatiquement la session dans un context manager."""
        self.close()
    
    def __repr__(self) -> str:
        """Représentation du client."""
        return f"LLMClient(base_url='{self.base_url}', timeout={self.timeout})"

