"""
Provider pour l'API locale (inférence sur GPUs AMD).
"""

from typing import List, Dict, Any, Optional
from src.clients.providers.base_provider import BaseProvider
from src.utils.models.message_model import MessageModel
from src.clients.base_api.base_api_client import BaseApiClient
from src.clients.base_api.endpoints.chats import ChatEndpoints
from src.clients.base_api.endpoints.models import ModelEndpoints
from src.utils.models_templates.templates_manager import TemplatesManager
from src.services.model_registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)


class LocalProvider(BaseProvider):
    """
    Provider pour l'API locale qui gère l'inférence sur GPUs AMD.
    
    Ce provider :
    - Formate les messages selon les templates (Mistral, Qwen, etc.)
    - Trouve le GPU qui héberge le modèle
    - Envoie les requêtes à l'API locale
    """
    
    def __init__(self, api_client: Optional[BaseApiClient] = None):
        """
        Initialise le provider local.
        
        Args:
            api_client: Client API locale (créé par défaut si None)
        """
        self.api_client = api_client or BaseApiClient()
        self.chat_endpoints = ChatEndpoints(self.api_client)
        self.model_endpoints = ModelEndpoints(self.api_client)
    
    @property
    def name(self) -> str:
        """
        Retourne le nom du provider.
        
        Returns:
            "local"
        """
        return "Local"
    
    def send_message(
        self,
        messages: List[MessageModel],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Envoie un message à l'API locale et retourne la réponse.
        
        Le provider formate les messages selon le template du modèle,
        trouve le GPU qui héberge le modèle, et envoie la requête.
        
        Args:
            messages: Liste de messages (contexte de la conversation)
            model_name: Nom du modèle à utiliser
            temperature: Température pour la génération (0.0-2.0)
            max_tokens: Nombre maximum de tokens à générer (défaut: 200)
            **kwargs: Paramètres additionnels (non utilisés pour l'instant)
        
        Returns:
            Contenu de la réponse de l'assistant (string)
        
        Raises:
            ValueError: Si le modèle n'est pas disponible ou si aucun GPU n'est trouvé
            ConnectionError: Si l'API locale n'est pas accessible
        """
        if not messages:
            raise ValueError("La liste de messages ne peut pas être vide")
        
        # Valeur par défaut pour max_tokens
        max_new_tokens = max_tokens if max_tokens is not None else 200
        
        # 1. Formater les messages selon le template du modèle
        # Convertir les MessageModel en format dict pour les templates
        messages_dict = MessageModel.messages_to_dict(messages)
        
        # Formater selon le template (Mistral, Qwen, etc.)
        formatted_context = TemplatesManager.format_messages(
            model_name,
            messages_dict
        )
        
        # 2. Extraire le dernier message utilisateur (nécessaire pour certains modèles comme Qwen)
        last_user_message = None
        for message in reversed(messages):
            if message.role.value == "user":
                last_user_message = message.content
                break
        
        # 3. Trouver le GPU ID qui héberge le modèle
        gpu_id = self._find_gpu_for_model(model_name)
        if gpu_id is None:
            raise ValueError(
                f"Aucun GPU trouvé avec le modèle '{model_name}' chargé. "
                f"Chargez d'abord le modèle sur un GPU avec POST /models/load."
            )
        
        # 4. Envoyer à l'API locale via ChatEndpoints
        try:
            response_data = self.chat_endpoints.post_chat(
                gpu_id=gpu_id,
                model_name=model_name,
                formatted_context=formatted_context,
                last_user_message=last_user_message,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi à l'API locale: {e}")
            raise ConnectionError(f"Impossible de communiquer avec l'API locale: {e}") from e
        
        # 5. Extraire la réponse (format peut varier selon l'API)
        assistant_content = (
            response_data.get("response", "") or 
            response_data.get("content", "") or
            response_data.get("text", "")
        )
        
        if not assistant_content:
            raise ValueError(f"Réponse vide de l'API locale: {response_data}")
        
        return assistant_content
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des modèles disponibles sur l'API locale.
        
        Returns:
            Liste de dictionnaires contenant les informations sur les modèles.
            Format : [{"identifier": "model-name", ...}, ...]
        
        Raises:
            ConnectionError: Si l'API locale n'est pas accessible
        """
        try:
            # Récupérer les modèles GGUF disponibles
            response = self.model_endpoints.get_models_gguf()
            
            # Le format de réponse peut varier, adapter selon votre API
            if isinstance(response, dict):
                # Si la réponse est un dict avec une clé 'models' ou 'data'
                models = response.get('models', response.get('data', []))
            elif isinstance(response, list):
                # Si la réponse est directement une liste
                models = response
            else:
                logger.warning(f"Format de réponse inattendu: {type(response)}")
                models = []
            
            # Formater les modèles au format attendu
            formatted_models = []
            for model in models:
                if isinstance(model, dict):
                    # Extraire l'identifiant (peut être 'name', 'identifier', 'id', etc.)
                    identifier = (
                        model.get('name') or 
                        model.get('identifier') or 
                        model.get('id') or 
                        model.get('alias', '')
                    )
                    
                    if identifier:
                        formatted_models.append({
                            "identifier": identifier,
                            "name": model.get('name', identifier),
                            "alias": model.get('alias', identifier),
                            # Ajouter d'autres champs si disponibles
                        })
            
            return formatted_models
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles: {e}")
            raise ConnectionError(f"Impossible de récupérer les modèles de l'API locale: {e}") from e
    
    def is_available(self) -> bool:
        """
        Vérifie si l'API locale est disponible et accessible.
        
        Returns:
            True si l'API locale est accessible, False sinon
        """
        try:
            # Tenter de se connecter à l'endpoint root
            response = self.api_client.root()
            return response is not None
        except Exception as e:
            logger.debug(f"API locale non disponible: {e}")
            return False
    
    def _find_gpu_for_model(self, model_name: str) -> Optional[int]:
        """
        Trouve le GPU ID qui héberge le modèle spécifié.
        
        Cette méthode interroge l'API /health pour trouver quel GPU
        a le modèle chargé.
        
        Args:
            model_name: Nom du modèle à chercher
        
        Returns:
            GPU ID si trouvé, None sinon
        """
        try:
            # Récupérer l'état de santé de tous les GPUs
            health_data = self.api_client.health()
            
            if not health_data or 'gpus' not in health_data:
                logger.warning("Aucune donnée GPU disponible dans la réponse /health")
                return None
            
            # Chercher dans tous les GPUs
            for gpu_id_str, gpu_info in health_data['gpus'].items():
                gpu_model_name = gpu_info.get('model', {}).get('name')
                if gpu_model_name == model_name:
                    return int(gpu_id_str)
            
            # Si non trouvé, essayer avec le nom normalisé (alias)
            normalized_name = ModelRegistry.get_full_name(model_name) or model_name
            for gpu_id_str, gpu_info in health_data['gpus'].items():
                gpu_model_name = gpu_info.get('model', {}).get('name', '')
                # Recherche partielle (le nom peut contenir des extensions)
                if normalized_name in gpu_model_name or model_name in gpu_model_name:
                    return int(gpu_id_str)
            
            logger.warning(f"Modèle '{model_name}' non trouvé sur aucun GPU")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche du GPU pour le modèle '{model_name}': {e}")
            return None
