"""
Endpoint pour la route racine de l'API.
"""
from typing import Dict, Any
from .base import BaseEndpoint


class RootEndpoint(BaseEndpoint):
    """
    Endpoint pour interagir avec la route racine de l'API.
    
    Permet de vérifier que l'API est active et accessible.
    """
    
    def check_status(self) -> Dict[str, Any]:
        """
        Vérifie que l'API est active.
        
        Effectue une requête GET vers la route racine pour vérifier
        que l'API répond correctement.
        
        Returns:
            Dictionnaire contenant le statut de l'API:
            {
                "message": "API Flask active",
                "status": "ok"
            }
        
        Raises:
            requests.HTTPError: Si l'API ne répond pas ou retourne une erreur
            requests.RequestException: Si la requête échoue (connexion, timeout, etc.)
        """
        return self.get("/")

