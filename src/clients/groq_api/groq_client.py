from groq import Groq
from groq._exceptions import APIError, APIConnectionError, APIStatusError, APITimeoutError
from typing import List, Dict, Optional
import logging

from src.clients.groq_api.config import GROQ_BASE_API_URL, GROQ_API_KEY
from src.clients.groq_api.endpoints.chat import send_messages
from src.clients.groq_api.endpoints.models import get_models

logger = logging.getLogger(__name__)


class GroqClientError(Exception):
    """Exception de base pour les erreurs Groq."""
    pass


class GroqAPIKeyError(GroqClientError):
    """Erreur de clé API Groq."""
    pass


class GroqRateLimitError(GroqClientError):
    """Erreur de rate limit Groq."""
    pass


class GroqConnectionError(GroqClientError):
    """Erreur de connexion à l'API Groq."""
    pass


class GroqTimeoutError(GroqClientError):
    """Erreur de timeout Groq."""
    pass


class GroqClient:

    def __init__(self):
        self.api_url = GROQ_BASE_API_URL
        self.api_key = GROQ_API_KEY
        if not self.api_key:
            logger.warning("Clé API Groq non configurée")
        self.client = Groq(api_key=self.api_key, base_url=self.api_url) if self.api_key else None

    def get_models(self):
        """Récupère les modèles disponibles avec gestion d'erreurs spécifique."""
        try:
            return get_models(self)
        except APIStatusError as e:
            if e.status_code == 401:
                raise GroqAPIKeyError("Clé API Groq invalide ou manquante") from e
            elif e.status_code == 429:
                raise GroqRateLimitError("Limite de taux dépassée pour l'API Groq") from e
            else:
                logger.error(f"Erreur API Groq (status {e.status_code}): {e}")
                raise GroqClientError(f"Erreur API Groq: {e}") from e
        except APIConnectionError as e:
            raise GroqConnectionError(f"Impossible de se connecter à l'API Groq: {e}") from e
        except APITimeoutError as e:
            raise GroqTimeoutError(f"Timeout lors de la connexion à l'API Groq: {e}") from e
        except APIError as e:
            logger.error(f"Erreur API Groq: {e}")
            raise GroqClientError(f"Erreur API Groq: {e}") from e
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération des modèles Groq: {e}")
            raise GroqClientError(f"Erreur inattendue: {e}") from e
    
    def send_messages(self, messages: List[Dict]):
        """Envoie des messages avec gestion d'erreurs spécifique."""
        try:
            return send_messages(self, messages)
        except APIStatusError as e:
            if e.status_code == 401:
                raise GroqAPIKeyError("Clé API Groq invalide ou manquante") from e
            elif e.status_code == 429:
                raise GroqRateLimitError("Limite de taux dépassée pour l'API Groq") from e
            else:
                logger.error(f"Erreur API Groq (status {e.status_code}): {e}")
                raise GroqClientError(f"Erreur API Groq: {e}") from e
        except APIConnectionError as e:
            raise GroqConnectionError(f"Impossible de se connecter à l'API Groq: {e}") from e
        except APITimeoutError as e:
            raise GroqTimeoutError(f"Timeout lors de la connexion à l'API Groq: {e}") from e
        except APIError as e:
            logger.error(f"Erreur API Groq: {e}")
            raise GroqClientError(f"Erreur API Groq: {e}") from e
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'envoi de messages Groq: {e}")
            raise GroqClientError(f"Erreur inattendue: {e}") from e

    
