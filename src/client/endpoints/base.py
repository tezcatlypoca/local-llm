"""
Classe de base pour tous les endpoints de l'API.
Fournit les méthodes communes pour les requêtes HTTP.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
import requests

if TYPE_CHECKING:
    from ..client import LLMClient

logger = logging.getLogger(__name__)


class BaseEndpoint:
    """
    Classe de base pour tous les endpoints de l'API.
    
    Fournit des méthodes utilitaires communes pour effectuer des requêtes HTTP
    vers l'API et gérer les erreurs de manière cohérente.
    """
    
    def __init__(self, client: LLMClient):
        """
        Initialise l'endpoint avec une référence au client principal.
        
        Args:
            client: Instance du client LLMClient pour accéder à la session HTTP
                   et à la configuration (base_url, etc.)
        """
        self.client = client
    
    @property
    def base_url(self) -> str:
        """Retourne l'URL de base de l'API depuis le client."""
        return self.client.base_url
    
    @property
    def session(self) -> requests.Session:
        """Retourne la session HTTP depuis le client."""
        return self.client.session
    
    def _build_url(self, path: str) -> str:
        """
        Construit une URL complète à partir d'un chemin relatif.
        
        Args:
            path: Chemin relatif (ex: "/models", "/chat/0")
        
        Returns:
            URL complète (ex: "http://localhost:5000/models")
        """
        # S'assurer que le path commence par /
        if not path.startswith('/'):
            path = '/' + path
        return f"{self.base_url.rstrip('/')}{path}"
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Gère la réponse HTTP et extrait le JSON.
        
        Args:
            response: Objet Response de requests
        
        Returns:
            Dictionnaire contenant les données JSON de la réponse
        
        Raises:
            requests.HTTPError: Si le code de statut HTTP indique une erreur
            ValueError: Si la réponse ne contient pas de JSON valide
        """
        try:
            # Tenter de parser le JSON
            data = response.json()
        except ValueError as e:
            logger.error(f"Erreur lors du parsing JSON de la réponse: {e}")
            logger.error(f"Contenu de la réponse: {response.text[:500]}")
            raise ValueError(f"Réponse invalide (non-JSON) de l'API: {response.text[:200]}")
        
        # Vérifier le code de statut HTTP
        if not response.ok:
            error_msg = data.get('message', f"Erreur HTTP {response.status_code}")
            logger.error(f"Erreur API {response.status_code}: {error_msg}")
            response.raise_for_status()  # Lève une exception HTTPError
        
        return data
    
    def get(
        self, 
        path: str, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une requête GET.
        
        Args:
            path: Chemin de l'endpoint (ex: "/models", "/health/0")
            params: Paramètres de requête (query parameters)
            **kwargs: Arguments additionnels à passer à requests.get()
        
        Returns:
            Dictionnaire contenant les données JSON de la réponse
        """
        url = self._build_url(path)
        logger.debug(f"GET {url} (params: {params})")
        
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.client.timeout,
                **kwargs
            )
            return self._handle_response(response)
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la requête GET {url}: {e}")
            raise
    
    def post(
        self, 
        path: str, 
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une requête POST.
        
        Args:
            path: Chemin de l'endpoint (ex: "/models/load/gpt2", "/chat/0")
            json: Données JSON à envoyer (sera sérialisées en JSON)
            data: Données à envoyer (alternative à json)
            **kwargs: Arguments additionnels à passer à requests.post()
        
        Returns:
            Dictionnaire contenant les données JSON de la réponse
        """
        url = self._build_url(path)
        logger.debug(f"POST {url} (json: {json is not None}, data: {data is not None})")
        
        try:
            response = self.session.post(
                url,
                json=json,
                data=data,
                timeout=self.client.timeout,
                **kwargs
            )
            return self._handle_response(response)
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la requête POST {url}: {e}")
            raise
    
    def put(
        self, 
        path: str, 
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une requête PUT.
        
        Args:
            path: Chemin de l'endpoint
            json: Données JSON à envoyer
            **kwargs: Arguments additionnels à passer à requests.put()
        
        Returns:
            Dictionnaire contenant les données JSON de la réponse
        """
        url = self._build_url(path)
        logger.debug(f"PUT {url}")
        
        try:
            response = self.session.put(
                url,
                json=json,
                timeout=self.client.timeout,
                **kwargs
            )
            return self._handle_response(response)
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la requête PUT {url}: {e}")
            raise
    
    def delete(
        self, 
        path: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une requête DELETE.
        
        Args:
            path: Chemin de l'endpoint
            **kwargs: Arguments additionnels à passer à requests.delete()
        
        Returns:
            Dictionnaire contenant les données JSON de la réponse
        """
        url = self._build_url(path)
        logger.debug(f"DELETE {url}")
        
        try:
            response = self.session.delete(
                url,
                timeout=self.client.timeout,
                **kwargs
            )
            return self._handle_response(response)
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la requête DELETE {url}: {e}")
            raise

