"""
Configuration pour le client API.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Charger les variables d'environnement depuis un fichier .env si présent
load_dotenv()


class APIConfig:
    """
    Configuration pour le client API.
    
    Les valeurs peuvent être définies via:
    1. Variables d'environnement (priorité)
    2. Paramètres passés au constructeur
    3. Valeurs par défaut
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        headers: Optional[dict] = None
    ):
        """
        Initialise la configuration.
        
        Args:
            base_url: URL de base de l'API (ex: "http://localhost:5000")
                     Si None, utilise la variable d'environnement LLM_API_BASE_URL
                     ou la valeur par défaut
            timeout: Timeout pour les requêtes HTTP en secondes
                     Si None, utilise la variable d'environnement LLM_API_TIMEOUT
                     ou la valeur par défaut (30)
            headers: Headers HTTP personnalisés à inclure dans toutes les requêtes
        """
        # Base URL
        self.base_url = (
            base_url or 
            os.getenv("LLM_API_BASE_URL") or 
            "http://192.168.1.50:5000"
        )
        
        # Timeout
        timeout_str = os.getenv("LLM_API_TIMEOUT")
        if timeout is None and timeout_str:
            try:
                timeout = int(timeout_str)
            except ValueError:
                timeout = 30
        self.timeout = timeout or 30
        
        # Headers par défaut
        self.headers = headers or {}
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"
    
    def __repr__(self) -> str:
        """Représentation de la configuration."""
        return (
            f"APIConfig(base_url='{self.base_url}', "
            f"timeout={self.timeout}, "
            f"headers={len(self.headers)} headers)"
        )

