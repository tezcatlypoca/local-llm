from src.utils.models.conversation_model import ConversationModel
from src.utils.models.message_model import MessageModel, Role
from src.services.model_registry import ModelRegistry
from src.services.context_manager import ContextManager
from src.repositories.conversation_repository import ConversationRepository
from src.clients.providers.provider_factory import ProviderFactory
from src.clients.providers.base_provider import BaseProvider
from typing import Optional, List
import logging
import os

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Gestionnaire de conversations - Logique métier uniquement.
    Ne stocke rien, délègue la persistance au Repository.
    """
    
    def __init__(self, 
                 repository: Optional[ConversationRepository] = None,
                 context_manager: Optional[ContextManager] = None):
        """
        Initialise le gestionnaire avec ses dépendances.
        
        Args:
            repository: Repository pour la persistance (créé par défaut si None)
            context_manager: Gestionnaire de contexte (créé par défaut si None)
        """
        self.repository = repository or ConversationRepository()
        self.context_manager = context_manager or ContextManager(self.repository)
    
    ########### Conversations functions ###########
    
    # POST conversations/create
    def create_conversation(self, 
                            model_name: str,
                            id: Optional[int] = 0, 
                            name: Optional[str] = "Conversation",
                            provider: Optional[str] = None,
                            temperature: Optional[float] = 0.7, 
                            message_max: Optional[int] = 10,
                            system_prompt: Optional[str] = None) -> ConversationModel:
        """
        Crée une nouvelle conversation.
        
        Args:
            model_name: Nom ou alias du modèle
            id: ID de la conversation (0 pour auto-généré)
            name: Nom de la conversation
            provider: Provider à utiliser ("local" ou "groq", défaut: "local" ou DEFAULT_API_PROVIDER)
            temperature: Température pour le modèle
            message_max: Nombre maximum de messages
        
        Returns:
            ConversationModel créée et sauvegardée
        """
        # Récupérer le nom complet du modèle via le registry
        full_model_name = ModelRegistry.get_full_name(model_name) or model_name
        
        # Déterminer le provider (paramètre > variable d'environnement > défaut "local")
        if provider is None:
            provider = os.getenv('DEFAULT_API_PROVIDER', 'local').lower()
        else:
            provider = provider.lower()
        
        # Valider le provider
        if provider not in ['local', 'groq']:
            raise ValueError(f"Provider '{provider}' non reconnu. Utilisez 'local' ou 'groq'")
        
        # Message système par défaut si non fourni (répondre en français et de manière concise)
        default_system_prompt = (
            "Tu es un assistant utile et concis. "
            "Réponds toujours en français. "
            "Sois direct et précis dans tes réponses. "
            "Évite les explications trop longues sauf si demandé explicitement."
        )
        final_system_prompt = system_prompt if system_prompt is not None else default_system_prompt
        
        # Créer l'objet conversation
        conv = ConversationModel(
            id=id, 
            name=name,
            model_name=full_model_name,
            provider=provider,
            temperature=temperature, 
            message_max=message_max,
            system_prompt=final_system_prompt
        )
        
        # Sauvegarder via le repository
        return self.repository.create(conv)
    
    # GET conversations/
    def get_conversations(self) -> List[ConversationModel]:
        """
        Récupère toutes les conversations.
        
        Returns:
            Liste de toutes les conversations
        """
        return self.repository.get_all()
    
    # GET conversations/{id}
    def get_conversation(self, id: int) -> Optional[ConversationModel]:
        """
        Récupère une conversation par son ID.
        
        Args:
            id: ID de la conversation
        
        Returns:
            ConversationModel ou None si non trouvée
        """
        return self.repository.get_by_id(id)
    
    # POST conversations/delete/{id}
    def delete_conversation(self, id: int) -> bool:
        """
        Supprime une conversation.
        
        Args:
            id: ID de la conversation à supprimer
        
        Returns:
            True si supprimée, False si non trouvée
        """
        return self.repository.delete(id)
    
    # DELETE conversations/
    def delete_all_conversations(self) -> int:
        """
        Supprime toutes les conversations et tous leurs messages.
        
        Returns:
            Nombre de conversations supprimées
        """
        return self.repository.delete_all()

    ########### Messages functions ###########

    # POST conversations/{id}/message
    def post_message(
        self, 
        conversation_id: int, 
        content: str,
        provider_override: Optional[str] = None
    ) -> MessageModel:
        """
        Envoie un message dans une conversation et récupère la réponse.
        
        Utilise le provider configuré dans la conversation, ou le provider_override
        si fourni (pour tests ou flexibilité).
        
        Args:
            conversation_id: ID de la conversation
            content: Contenu du message utilisateur
            provider_override: Provider à utiliser pour ce message (optionnel, surcharge conversation.provider)
        
        Returns:
            MessageModel de la réponse de l'assistant
        
        Raises:
            ValueError: Si la conversation n'existe pas ou si le provider est invalide
        """
        # Vérifier que la conversation existe
        conversation = self.repository.get_by_id(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} n'existe pas")
        
        # Créer le message utilisateur
        user_message = MessageModel(
            id=0,  # Auto-généré par le repository
            conversation_id=conversation_id,
            role=Role.USER,
            content=content
        )
        
        # Sauvegarder le message utilisateur (non formaté)
        self.repository.add_message(conversation_id, user_message)
        
        # Recharger la conversation pour avoir tous les messages (y compris celui qu'on vient d'ajouter)
        conversation = self.repository.get_by_id(conversation_id)
        
        # Déterminer le provider à utiliser (surcharge > conversation.provider)
        provider_name = provider_override.lower() if provider_override else conversation.provider.lower()
        
        # Obtenir le provider via la factory
        try:
            provider = ProviderFactory.get_provider(provider_name)
        except ValueError as e:
            raise ValueError(f"Provider invalide: {e}") from e
        
        # Préparer les messages pour le provider
        # Pour LocalProvider : besoin de troncature (gérée par ContextManager)
        # Pour GroqProvider : besoin des messages bruts (tronqués si nécessaire)
        
        # Préparer les messages avec le message système si configuré
        messages_to_send = conversation.message.copy()
        
        # Ajouter le message système au début s'il est configuré et qu'il n'existe pas déjà
        if conversation.system_prompt:
            # Vérifier si un message système existe déjà
            has_system_message = any(msg.role == Role.SYSTEM for msg in messages_to_send)
            if not has_system_message:
                system_message = MessageModel(
                    id=0,
                    conversation_id=conversation_id,
                    role=Role.SYSTEM,
                    content=conversation.system_prompt
                )
                messages_to_send.insert(0, system_message)
        
        # Tronquer les messages si nécessaire (commun aux deux providers)
        # Utiliser la méthode publique de ContextManager qui gère la troncature
        truncated_messages = self.context_manager.truncate_messages_if_needed(
            messages_to_send,
            conversation.model_name
        )
        
        # Envoyer via le provider
        try:
            assistant_content = provider.send_message(
                messages=truncated_messages,
                model_name=conversation.model_name,
                temperature=conversation.temperature,
                max_tokens=1024  # Augmenté pour permettre des réponses plus longues (les balises de raisonnement consomment aussi des tokens)
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi via le provider '{provider_name}': {e}")
            raise
        
        if not assistant_content:
            raise ValueError(f"Réponse vide du provider '{provider_name}'")
        
        # Créer le message assistant
        assistant_message = MessageModel(
            id=0,  # Auto-généré par le repository
            conversation_id=conversation_id,
            role=Role.ASSISTANT,
            content=assistant_content
        )
        
        # Sauvegarder la réponse
        self.repository.add_message(conversation_id, assistant_message)
        
        return assistant_message

    def get_messages(self, conversation_id: int) -> List[MessageModel]:
        return self.repository.get_messages(conversation_id)
    
    def get_message(self, conversation_id: int, message_id: int) -> Optional[MessageModel]:
        return self.repository.get_message(conversation_id, message_id)
    
    def delete_message(self, conversation_id: int, message_id: int) -> bool:
        return self.repository.delete_message(conversation_id, message_id)