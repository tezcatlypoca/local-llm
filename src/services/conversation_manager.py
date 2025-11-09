from src.utils.models.conversation_model import ConversationModel
from src.utils.models.message_model import MessageModel, Role
from src.services.model_registry import ModelRegistry
from src.services.context_manager import ContextManager
from src.repositories.conversation_repository import ConversationRepository
from src.clients.base_api_client import BaseApiClient
from src.clients.endpoints.chats import ChatEndpoints
from src.utils.models_templates.templates_manager import TemplatesManager
from typing import Optional, List

class ConversationManager:
    """
    Gestionnaire de conversations - Logique métier uniquement.
    Ne stocke rien, délègue la persistance au Repository.
    """
    
    def __init__(self, 
                 repository: Optional[ConversationRepository] = None,
                 context_manager: Optional[ContextManager] = None,
                 api_client: Optional[BaseApiClient] = None):
        """
        Initialise le gestionnaire avec ses dépendances.
        
        Args:
            repository: Repository pour la persistance (créé par défaut si None)
            context_manager: Gestionnaire de contexte (créé par défaut si None)
            api_client: Client API pour envoyer les messages (créé par défaut si None)
        """
        self.repository = repository or ConversationRepository()
        self.context_manager = context_manager or ContextManager(self.repository)
        self.api_client = api_client or BaseApiClient()
        self.chat_endpoints = ChatEndpoints(self.api_client)
    
    ########### Conversations functions ###########
    
    # POST conversations/create
    def create_conversation(self, 
                            model_name: str,
                            id: Optional[int] = 0, 
                            name: Optional[str] = "Conversation",
                            temperature: Optional[float] = 0.7, 
                            message_max: Optional[int] = 10) -> ConversationModel:
        """
        Crée une nouvelle conversation.
        
        Args:
            model_name: Nom ou alias du modèle
            id: ID de la conversation (0 pour auto-généré)
            name: Nom de la conversation
            temperature: Température pour le modèle
            message_max: Nombre maximum de messages
        
        Returns:
            ConversationModel créée et sauvegardée
        """
        # Récupérer le nom complet du modèle via le registry
        full_model_name = ModelRegistry.get_full_name(model_name) or model_name
        
        # Créer l'objet conversation
        conv = ConversationModel(
            id=id, 
            name=name, 
            model_name=full_model_name, 
            temperature=temperature, 
            message_max=message_max
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

    ########### Messages functions ###########

    # POST conversations/{id}/message
    def post_message(self, conversation_id: int, content: str) -> MessageModel:
        """
        Envoie un message dans une conversation et récupère la réponse.
        
        Args:
            conversation_id: ID de la conversation
            content: Contenu du message utilisateur
        
        Returns:
            MessageModel de la réponse de l'assistant
        
        Raises:
            ValueError: Si la conversation n'existe pas
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
        
        # Construire le contexte formaté pour l'API
        formatted_context = self.context_manager.build_context(conversation_id)
        
        # Envoyer à l'API externe
        # TODO: Adapter le format selon ce que l'API attend
        response_data = self.chat_endpoints.post_chat({
            "message": formatted_context,
            "temperature": conversation.temperature
        })
        
        # Extraire la réponse (à adapter selon le format de l'API)
        assistant_content = response_data.get("response", "") or response_data.get("content", "")
        
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

    