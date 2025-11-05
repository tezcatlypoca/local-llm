"""
Client pour appeler l'API de base via LLMClient.
"""
import logging
import os
from typing import Optional
from dotenv import load_dotenv
from ..client import LLMClient

# Charger les variables d'environnement
load_dotenv()

logger = logging.getLogger(__name__)


def get_base_api_client() -> LLMClient:
    """
    Crée et retourne une instance de LLMClient configurée pour l'API de base.
    
    Utilise la variable d'environnement BASE_API_URL ou LLM_API_BASE_URL,
    ou la valeur par défaut http://localhost:5000
    """
    base_url = (
        os.getenv("BASE_API_URL") or
        os.getenv("LLM_API_BASE_URL") or
        "http://localhost:5000"
    )
    
    return LLMClient(base_url=base_url)


# Instance globale du client
base_api_client = get_base_api_client()
