"""
Provider pour l'API Groq (inférence cloud rapide).
"""

from typing import List, Dict, Any, Optional
import re
from src.clients.providers.base_provider import BaseProvider
from src.utils.models.message_model import MessageModel
from src.clients.groq_api.groq_client import (
    GroqClient,
    GroqClientError,
    GroqAPIKeyError,
    GroqRateLimitError,
    GroqConnectionError,
    GroqTimeoutError
)
from src.clients.groq_api.endpoints.models import get_models
import logging

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    """
    Provider pour l'API Groq qui offre une inférence cloud rapide.
    
    Ce provider :
    - Convertit les messages au format Groq (array de messages)
    - Envoie les requêtes à l'API Groq
    - Gère l'authentification via API key
    """
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        """
        Initialise le provider Groq.
        
        Args:
            groq_client: Client Groq (créé par défaut si None)
        """
        self.groq_client = groq_client or GroqClient()
    
    @property
    def name(self) -> str:
        """
        Retourne le nom du provider.
        
        Returns:
            "Groq"
        """
        return "Groq"
    
    def send_message(
        self,
        messages: List[MessageModel],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Envoie un message à l'API Groq et retourne la réponse.
        
        Groq utilise le format OpenAI avec un array de messages.
        Pas besoin de formatage de template, juste conversion en dict.
        
        Args:
            messages: Liste de messages (contexte de la conversation)
            model_name: Nom du modèle à utiliser (ex: "mixtral-8x7b-32768")
            temperature: Température pour la génération (0.0-2.0)
            max_tokens: Nombre maximum de tokens à générer (optionnel)
            **kwargs: Paramètres additionnels (is_stream, etc.)
        
        Returns:
            Contenu de la réponse de l'assistant (string)
        
        Raises:
            ValueError: Si le modèle n'est pas disponible ou si les paramètres sont invalides
            ConnectionError: Si l'API Groq n'est pas accessible
        """
        if not messages:
            raise ValueError("La liste de messages ne peut pas être vide")
        
        # 1. Convertir les MessageModel en format dict pour Groq
        # Groq utilise le format OpenAI : [{"role": "user", "content": "..."}, ...]
        messages_dict = self._messages_to_dict_list(messages)
        
        # 2. Gérer le message système pour les modèles qui ne le supportent pas bien
        # Certains modèles (comme Qwen) peuvent ne pas bien respecter le rôle "system"
        # Dans ce cas, on intègre le message système dans le premier message user
        system_message = None
        if messages_dict and messages_dict[0].get('role') == 'system':
            system_message = messages_dict.pop(0)  # Retirer le message système de la liste
            
            # Si le modèle est Qwen, intégrer le système dans le premier message user
            # Sinon, remettre le système en premier (Groq supporte généralement "system")
            if 'qwen' in model_name.lower():
                # Pour Qwen, intégrer le système dans le premier message user
                if messages_dict and messages_dict[0].get('role') == 'user':
                    messages_dict[0]['content'] = f"{system_message['content']}\n\n{messages_dict[0]['content']}"
                    logger.debug(f"Message système intégré dans le premier message user pour Qwen")
                else:
                    # Pas de message user, remettre le système
                    messages_dict.insert(0, system_message)
            else:
                # Pour les autres modèles, remettre le système en premier
                messages_dict.insert(0, system_message)
                logger.debug(f"Message système conservé en premier pour {model_name}")
        
        # Log pour déboguer : vérifier que le message système est présent ou intégré
        has_system = any(msg.get('role') == 'system' for msg in messages_dict)
        if has_system or system_message:
            logger.debug(f"Message système géré pour {model_name}: présent={has_system}, intégré={system_message is not None and not has_system}")
        else:
            logger.warning("Aucun message système trouvé dans les messages envoyés à Groq")
        
        # 2. Vérifier que le modèle est disponible (optionnel, peut être coûteux)
        # On peut le désactiver pour améliorer les performances
        # if not self.validate_model(model_name):
        #     raise ValueError(f"Modèle '{model_name}' non disponible sur Groq")
        
        # 3. Extraire les paramètres optionnels
        is_stream = kwargs.get('is_stream', False)
        
        # 4. Envoyer à l'API Groq via le client
        try:
            # Utiliser directement groq_client.client pour avoir plus de contrôle
            # et supporter max_tokens si fourni
            params = {
                "messages": messages_dict,
                "model": model_name,
                "temperature": temperature,
                "stream": is_stream
            }
            
            # Ajouter max_tokens si fourni (Groq supporte ce paramètre)
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Appeler l'API Groq directement
            response = self.groq_client.client.chat.completions.create(**params)
            
            # Extraire le contenu de la réponse
            response_content = response.choices[0].message.content
            
            # Nettoyer la réponse : supprimer les balises de raisonnement (thinking tags)
            # Certains modèles (comme Qwen) incluent des balises comme <think>...</think>
            # Ces balises consomment des tokens, donc on les supprime pour ne garder que la réponse utile
            cleaned_content = self._clean_thinking_tags(response_content)
            
            return cleaned_content
            
        except GroqAPIKeyError as e:
            logger.error(f"Erreur de clé API Groq: {e}")
            raise ConnectionError(
                "Clé API Groq invalide ou manquante. "
                "Vérifiez la variable d'environnement GROQ_API_KEY."
            ) from e
        except GroqRateLimitError as e:
            logger.warning(f"Rate limit Groq atteint: {e}")
            raise ConnectionError(
                "Limite de taux dépassée pour l'API Groq. Réessayez plus tard."
            ) from e
        except GroqConnectionError as e:
            logger.error(f"Erreur de connexion Groq: {e}")
            raise ConnectionError(f"Impossible de se connecter à l'API Groq: {e}") from e
        except GroqTimeoutError as e:
            logger.error(f"Timeout Groq: {e}")
            raise ConnectionError(f"Timeout lors de la connexion à l'API Groq: {e}") from e
        except GroqClientError as e:
            logger.error(f"Erreur client Groq: {e}")
            raise ConnectionError(f"Erreur API Groq: {e}") from e
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'envoi à l'API Groq: {e}")
            raise ConnectionError(f"Erreur inattendue: {e}") from e
    
    @staticmethod
    def _clean_thinking_tags(content: str) -> str:
        """
        Nettoie la réponse en supprimant les balises de raisonnement (thinking tags).
        
        Certains modèles (comme Qwen) incluent des balises de raisonnement dans leurs réponses :
        - <think>...</think>
        - <think>...</think>
        - <reasoning>...</reasoning>
        - Et autres variantes
        
        Cette fonction supprime uniquement le contenu entre les balises, en gardant tout ce qui est en dehors.
        
        Args:
            content: Contenu brut de la réponse du modèle
        
        Returns:
            Contenu nettoyé sans les balises de raisonnement et leur contenu
        """
        if not content:
            return content
        
        # Liste des patterns de balises de raisonnement à supprimer
        # Format: <tag_name>...</tag_name> (on supprime la balise ET son contenu)
        # On garde tout ce qui est EN DEHORS des balises
        thinking_tag_patterns = [
            # Balise principale utilisée par Qwen (avec ou sans attributs)
            r'<redacted_reasoning[^>]*>.*?</think>',
            # Autres variantes possibles
            r'<think>.*?</think>',
            r'<think[^>]*>.*?</think>',
            r'<reasoning>.*?</reasoning>',
            r'<reasoning[^>]*>.*?</reasoning>',
            r'<thinking>.*?</thinking>',
            r'<thought>.*?</thought>',
            r'<internal>.*?</internal>',
        ]
        
        cleaned = content
        original_length = len(cleaned)
        for pattern in thinking_tag_patterns:
            # Utiliser re.DOTALL pour que . corresponde aussi aux sauts de ligne
            # Utiliser re.IGNORECASE pour ignorer la casse
            before_len = len(cleaned)
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            if len(cleaned) < before_len:
                logger.debug(f"Pattern '{pattern}' a supprimé {before_len - len(cleaned)} caractères")
        
        if len(cleaned) < original_length:
            logger.debug(f"Nettoyage: {original_length} → {len(cleaned)} caractères")
        
        # Nettoyer les espaces multiples et sauts de ligne en début/fin uniquement
        # Garder le contenu intact entre les balises supprimées
        cleaned = cleaned.strip()
        
        # Nettoyer les sauts de ligne multiples consécutifs (max 2) pour éviter trop d'espaces
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Si après nettoyage il ne reste rien, retourner le contenu original
        # (au cas où le pattern aurait tout supprimé par erreur)
        if not cleaned:
            logger.warning("Le nettoyage des balises de raisonnement a supprimé tout le contenu. Retour du contenu original.")
            return content
        
        return cleaned
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des modèles disponibles sur Groq.
        
        Returns:
            Liste de dictionnaires contenant les informations sur les modèles.
            Format : [{"identifier": "model-name", "context_window": ..., ...}, ...]
        
        Raises:
            ConnectionError: Si l'API Groq n'est pas accessible
        """
        try:
            # Utiliser la fonction get_models depuis endpoints
            models = get_models(self.groq_client)
            
            # Le format est déjà correct (retourné par _format_models_response)
            # Format : [{"identifier": "...", "context_window": ..., "max_completion_tokens": ...}, ...]
            return models
            
        except GroqAPIKeyError as e:
            logger.error(f"Erreur de clé API Groq: {e}")
            raise ConnectionError(
                "Clé API Groq invalide ou manquante. "
                "Vérifiez la variable d'environnement GROQ_API_KEY."
            ) from e
        except GroqRateLimitError as e:
            logger.warning(f"Rate limit Groq atteint: {e}")
            raise ConnectionError(
                "Limite de taux dépassée pour l'API Groq. Réessayez plus tard."
            ) from e
        except GroqConnectionError as e:
            logger.error(f"Erreur de connexion Groq: {e}")
            raise ConnectionError(f"Impossible de se connecter à l'API Groq: {e}") from e
        except GroqTimeoutError as e:
            logger.error(f"Timeout Groq: {e}")
            raise ConnectionError(f"Timeout lors de la connexion à l'API Groq: {e}") from e
        except GroqClientError as e:
            logger.error(f"Erreur client Groq: {e}")
            raise ConnectionError(f"Erreur API Groq: {e}") from e
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération des modèles Groq: {e}")
            raise ConnectionError(f"Erreur inattendue: {e}") from e
    
    def is_available(self) -> bool:
        """
        Vérifie si l'API Groq est disponible et accessible.
        
        Returns:
            True si l'API Groq est accessible, False sinon
        """
        try:
            # Vérifier que la clé API est configurée
            if not self.groq_client.api_key:
                logger.debug("Clé API Groq non configurée")
                return False
            
            # Tenter de récupérer les modèles (test de connexion)
            # On limite à 1 pour éviter de charger tous les modèles
            models = self.get_available_models()
            return len(models) > 0
            
        except Exception as e:
            logger.debug(f"API Groq non disponible: {e}")
            return False
