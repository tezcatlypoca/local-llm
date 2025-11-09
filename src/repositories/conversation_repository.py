from typing import List, Optional
from src.utils.models.conversation_model import ConversationModel
from src.utils.models.message_model import MessageModel


class ConversationRepository:
    """
    Repository pour la persistance des conversations et messages.
    Interface pour l'accès aux données (BDD).
    """
    
    def __init__(self):
        # TODO: Initialiser la connexion à la BDD ici
        # Pour l'instant, stockage en mémoire (sera remplacé par une vraie BDD)
        self._conversations: dict[int, ConversationModel] = {}
        self._messages: dict[int, List[MessageModel]] = {}  # conversation_id -> messages
        self._next_conversation_id = 1
        self._next_message_id = 1
    
    def create(self, conversation: ConversationModel) -> ConversationModel:
        """
        Crée une nouvelle conversation en BDD.
        
        Args:
            conversation: ConversationModel à sauvegarder
        
        Returns:
            ConversationModel avec l'ID généré
        """
        # Générer un ID si non fourni
        if conversation.id == 0:
            conversation.id = self._next_conversation_id
            self._next_conversation_id += 1
        
        # Sauvegarder
        self._conversations[conversation.id] = conversation
        self._messages[conversation.id] = []
        
        return conversation
    
    def get_by_id(self, id: int) -> Optional[ConversationModel]:
        """
        Récupère une conversation et tous ses messages depuis la BDD.
        
        Args:
            id: ID de la conversation
        
        Returns:
            ConversationModel avec ses messages, ou None si non trouvée
        """
        conversation = self._conversations.get(id)
        if conversation:
            # Charger les messages associés
            conversation.message = self._messages.get(id, [])
        return conversation
    
    def get_all(self) -> List[ConversationModel]:
        """
        Récupère toutes les conversations depuis la BDD.
        
        Returns:
            Liste de toutes les conversations
        """
        conversations = []
        for conv_id in self._conversations.keys():
            conv = self.get_by_id(conv_id)
            if conv:
                conversations.append(conv)
        return conversations
    
    def update(self, conversation: ConversationModel) -> ConversationModel:
        """
        Met à jour une conversation existante.
        
        Args:
            conversation: ConversationModel à mettre à jour
        
        Returns:
            ConversationModel mis à jour
        """
        if conversation.id in self._conversations:
            self._conversations[conversation.id] = conversation
        return conversation
    
    def delete(self, id: int) -> bool:
        """
        Supprime une conversation et tous ses messages.
        
        Args:
            id: ID de la conversation à supprimer
        
        Returns:
            True si supprimée, False si non trouvée
        """
        if id in self._conversations:
            del self._conversations[id]
            if id in self._messages:
                del self._messages[id]
            return True
        return False
    
    def add_message(self, conversation_id: int, message: MessageModel) -> MessageModel:
        """
        Ajoute un message à une conversation.
        
        Args:
            conversation_id: ID de la conversation
            message: MessageModel à ajouter
        
        Returns:
            MessageModel avec l'ID généré
        """
        # Générer un ID si non fourni
        if message.id == 0:
            message.id = self._next_message_id
            self._next_message_id += 1
        
        # S'assurer que la conversation existe
        if conversation_id not in self._conversations:
            raise ValueError(f"Conversation {conversation_id} n'existe pas")
        
        # Ajouter le message
        if conversation_id not in self._messages:
            self._messages[conversation_id] = []
        
        self._messages[conversation_id].append(message)
        
        return message
    
    def get_messages(self, conversation_id: int) -> List[MessageModel]:
        """
        Récupère tous les messages d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
        
        Returns:
            Liste des messages
        """
        return self._messages.get(conversation_id, [])

