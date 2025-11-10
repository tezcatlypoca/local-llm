import requests
from typing import Dict, Any, Optional
from src.utils.models_templates.templates_manager import TemplatesManager
import logging

logger = logging.getLogger(__name__)


class ChatEndpoints:

    def __init__(self, base_client):
        self.base_url = f"{base_client.get_base_url}/chat"
        self.base_client = base_client
    
    def _format_payload_for_model(
        self, 
        model_name: str, 
        formatted_context: str, 
        last_user_message: Optional[str],
        temperature: float = 0.7,
        max_new_tokens: int = 200
    ) -> Dict[str, Any]:
        """
        Formate le payload selon le format attendu par l'API pour chaque modèle.
        
        Args:
            model_name: Nom du modèle (normalisé ou complet)
            formatted_context: Contexte formaté selon le template du modèle
            last_user_message: Dernier message utilisateur (pour Qwen)
            temperature: Température pour la génération
            max_new_tokens: Nombre maximum de tokens à générer
        
        Returns:
            Dictionnaire formaté selon le modèle
        """
        normalized_name = TemplatesManager._normalize_model_name(model_name)
        
        if normalized_name == 'mistral':
            # Format Mistral : message formaté complet + temperature + max_new_tokens
            return {
                "message": formatted_context,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens
            }
        elif normalized_name == 'qwen':
            # Format Qwen : role + prompt (juste le dernier message utilisateur)
            # Note: Qwen semble attendre juste le dernier message, pas tout le contexte
            return {
                "role": "user",
                "prompt": last_user_message or formatted_context
            }
        else:
            # Format par défaut (compatible avec l'ancien format)
            logger.warning(f"Modèle '{model_name}' non reconnu, utilisation du format par défaut")
            return {
                "message": formatted_context,
                "temperature": temperature
            }
    
    def post_chat(
        self, 
        gpu_id: int,
        model_name: str,
        formatted_context: str,
        last_user_message: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 200
    ) -> Dict[str, Any]:
        """
        Envoie un message à l'API de chat.
        
        Args:
            gpu_id: ID du GPU sur lequel le modèle est chargé
            model_name: Nom du modèle (pour déterminer le format)
            formatted_context: Contexte formaté selon le template
            last_user_message: Dernier message utilisateur (pour Qwen)
            temperature: Température pour la génération
            max_new_tokens: Nombre maximum de tokens à générer
        
        Returns:
            Réponse de l'API au format JSON
        """
        # Formater le payload selon le modèle
        payload = self._format_payload_for_model(
            model_name=model_name,
            formatted_context=formatted_context,
            last_user_message=last_user_message,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        # Envoyer à l'API avec le gpu_id dans l'URL
        # S'assurer que l'URL a le protocole http://
        base_url = self.base_url
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        
        url = f"{base_url}/{gpu_id}"
        logger.debug(f"Envoi à {url} avec payload: {payload}")
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Lève une exception si erreur HTTP
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de l'envoi à l'API: {e}")
            raise
    