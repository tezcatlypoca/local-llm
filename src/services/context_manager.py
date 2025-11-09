from src.utils.models_templates.templates_manager import TemplatesManager
from src.utils.models.message_model import MessageModel, Role
from src.repositories.conversation_repository import ConversationRepository
from typing import List, Dict, Optional
import logging

# Import conditionnel pour transformers (peut ne pas être installé)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers non installé. Le comptage de tokens ne fonctionnera pas.")

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Gestionnaire de contexte pour formater les messages avant envoi à l'API.
    Récupère les conversations depuis le Repository.
    Gère le comptage et la troncature de tokens.
    """
    
    # Mapping des noms de modèles vers les tokenizers HuggingFace
    MODEL_TOKENIZER_MAP = {
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
        'qwen': 'Qwen/Qwen2-7B-Instruct',
    }
    
    def __init__(self, repository: ConversationRepository, max_context_length_tokens: int = 10000):
        """
        Initialise le gestionnaire de contexte.
        
        Args:
            repository: Repository pour récupérer les conversations et messages
            max_context_length_tokens: Nombre maximum de tokens dans le contexte
        """
        self.repository = repository
        self.max_context_length_tokens = max_context_length_tokens
        
        # Cache des tokenizers (évite de les recharger à chaque fois)
        self._tokenizers: Dict[str, 'AutoTokenizer'] = {}
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "transformers non disponible. Le comptage de tokens utilisera une estimation approximative."
            )
    
    def build_context(self, conversation_id: int) -> str:
        """
        Construit le contexte formaté pour une conversation.
        Utilise le cache si disponible pour éviter de reformater.
        
        Args:
            conversation_id: ID de la conversation
        
        Returns:
            Chaîne formatée selon le template du modèle
        
        Raises:
            ValueError: Si la conversation n'existe pas
        """
        # Récupérer la conversation et ses messages depuis le repository
        conversation = self.repository.get_by_id(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} n'existe pas")
        
        # Vérifier si le cache est valide
        current_message_count = len(conversation.message)
        if (conversation._formatted_cache is not None and 
            conversation._cache_version == current_message_count):
            # Cache valide, retourner directement
            return conversation._formatted_cache
        
        # Cache invalide ou inexistant, reformater
        # Tronquer les messages bruts AVANT de formater (plus efficace)
        truncated_messages = self._truncate_messages_if_needed(
            conversation.message,
            conversation.model_name
        )
        
        # Convertir en format Dict pour les templates
        messages_dict = MessageModel.messages_to_dict(truncated_messages)
        
        # Formater uniquement les messages tronqués
        formatted_context = TemplatesManager.format_messages(
            conversation.model_name,
            messages_dict
        )
        
        # Mettre à jour le cache
        conversation._formatted_cache = formatted_context
        conversation._cache_version = current_message_count
        
        # Sauvegarder le cache dans le repository
        self.repository.update(conversation)
        
        return formatted_context
    
    def _get_tokenizer(self, model_name: str) -> Optional['AutoTokenizer']:
        """
        Récupère ou crée le tokenizer pour un modèle donné.
        Utilise un cache pour éviter de recharger les tokenizers.
        
        Args:
            model_name: Nom du modèle (peut être un nom complet ou normalisé)
        
        Returns:
            AutoTokenizer ou None si transformers n'est pas disponible
        """
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        # Normaliser le nom du modèle
        normalized_name = TemplatesManager._normalize_model_name(model_name)
        
        # Vérifier le cache
        if normalized_name in self._tokenizers:
            return self._tokenizers[normalized_name]
        
        # Récupérer le nom du tokenizer HuggingFace
        hf_model_name = self.MODEL_TOKENIZER_MAP.get(normalized_name)
        
        if not hf_model_name:
            logger.warning(
                f"Tokenizer non trouvé pour le modèle '{model_name}' (normalisé: '{normalized_name}'). "
                f"Modèles supportés: {list(self.MODEL_TOKENIZER_MAP.keys())}"
            )
            return None
        
        try:
            # Charger le tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self._tokenizers[normalized_name] = tokenizer
            logger.info(f"Tokenizer chargé pour le modèle '{normalized_name}'")
            return tokenizer
        except Exception as e:
            logger.error(f"Erreur lors du chargement du tokenizer pour '{hf_model_name}': {e}")
            return None
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Compte le nombre de tokens dans un texte pour un modèle donné.
        
        Args:
            text: Texte à compter
            model_name: Nom du modèle
        
        Returns:
            Nombre de tokens
        """
        tokenizer = self._get_tokenizer(model_name)
        
        if tokenizer is None:
            # Estimation approximative si le tokenizer n'est pas disponible
            # ~2 tokens par mot en français, ~1.3 en anglais
            word_count = len(text.split())
            return word_count * 2  # Estimation conservatrice pour le français
        
        try:
            # Encoder le texte en tokens
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.error(f"Erreur lors du comptage de tokens: {e}")
            # Fallback sur l'estimation
            word_count = len(text.split())
            return word_count * 2
    
    def _truncate_if_needed(
        self, 
        formatted_context: str, 
        model_name: str,
        messages: List[MessageModel]
    ) -> str:
        """
        Tronque le contexte si il dépasse la limite de tokens.
        Stratégie : Garde les messages les plus récents (les plus importants).
        
        Args:
            formatted_context: Contexte formaté
            model_name: Nom du modèle
            messages: Liste des messages (pour troncature intelligente)
        
        Returns:
            Contexte tronqué si nécessaire
        """
        token_count = self.count_tokens(formatted_context, model_name)
        
        if token_count <= self.max_context_length_tokens:
            return formatted_context
        
        logger.warning(
            f"Contexte trop long ({token_count} tokens > {self.max_context_length_tokens}). "
            f"Troncature nécessaire."
        )
        
        # Stratégie de troncature : garder les messages les plus récents
        # On garde au minimum le dernier message user et le dernier assistant
        # Puis on ajoute progressivement les messages précédents jusqu'à la limite
        
        # Trier les messages par ordre chronologique (déjà dans l'ordre normalement)
        # On va retirer les messages les plus anciens
        
        # Calculer combien de messages on peut garder
        # On commence par garder les 2 derniers messages (user + assistant)
        min_messages_to_keep = 2
        
        for num_messages in range(min_messages_to_keep, len(messages) + 1):
            # Prendre les N derniers messages
            recent_messages = messages[-num_messages:]
            
            # Reformater uniquement ces messages
            messages_dict = MessageModel.messages_to_dict(recent_messages)
            truncated_context = TemplatesManager.format_messages(model_name, messages_dict)
            
            # Vérifier si ça rentre dans la limite
            truncated_token_count = self.count_tokens(truncated_context, model_name)
            
            if truncated_token_count <= self.max_context_length_tokens:
                # Si on peut ajouter un message de plus, on le fait
                if num_messages < len(messages):
                    # Essayer avec un message de plus
                    next_messages = messages[-(num_messages + 1):]
                    next_dict = MessageModel.messages_to_dict(next_messages)
                    next_context = TemplatesManager.format_messages(model_name, next_dict)
                    next_token_count = self.count_tokens(next_context, model_name)
                    
                    if next_token_count <= self.max_context_length_tokens:
                        continue  # On peut garder plus de messages
                
                # On a trouvé le bon nombre de messages
                logger.info(
                    f"Contexte tronqué : {len(messages)} messages → {num_messages} messages "
                    f"({truncated_token_count} tokens)"
                )
                return truncated_context
        
        # Si même avec 2 messages c'est trop long, on garde quand même
        # (le dernier message user doit être envoyé)
        # Prendre au minimum les 2 derniers messages
        min_messages = messages[-min_messages_to_keep:]
        min_dict = MessageModel.messages_to_dict(min_messages)
        min_context = TemplatesManager.format_messages(model_name, min_dict)
        
        logger.warning(
            f"Même avec {min_messages_to_keep} messages, le contexte dépasse la limite. "
            f"Envoi quand même (peut échouer côté API)."
        )
        return min_context

