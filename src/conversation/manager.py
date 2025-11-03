"""
Gestionnaire principal de conversations.
"""
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime

from ..client import LLMClient
from .context import ConversationContext, Message, MessageRole
from .templates.registry import TemplateRegistry
from .storage.base import StorageBackend
from .storage.sqlite_storage import SQLiteStorage
from .exceptions import ConversationNotFoundError


class ConversationManager:
    """
    Gestionnaire principal de conversations.
    
    Point d'entrée pour :
    - Créer/gérer des conversations
    - Envoyer des messages avec formatage automatique
    - Gérer le contexte
    - Persister les échanges
    """
    
    def __init__(
        self,
        client: LLMClient,
        storage: Optional[StorageBackend] = None,
        auto_save: bool = True,
        default_max_messages: Optional[int] = 50
    ):
        """
        Initialise le gestionnaire de conversations.
        
        Args:
            client: Client API pour communiquer avec l'API de base
            storage: Backend de stockage (par défaut SQLite)
            auto_save: Si True, sauvegarde automatiquement après chaque message
            default_max_messages: Nombre par défaut de messages à garder dans le contexte
        """
        self.client = client
        self.storage = storage or SQLiteStorage()
        self.auto_save = auto_save
        self.default_max_messages = default_max_messages
        self._active_conversations: Dict[str, ConversationContext] = {}
    
    def create_conversation(
        self,
        model_name: str,
        gpu_id: int,
        system_prompt: Optional[str] = None,
        max_messages: Optional[int] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Crée une nouvelle conversation.
        
        Args:
            model_name: Nom du modèle à utiliser
            gpu_id: ID du GPU sur lequel le modèle est chargé
            system_prompt: Prompt système optionnel
            max_messages: Nombre maximum de messages à garder (None = utiliser la valeur par défaut)
            conversation_id: ID personnalisé (généré automatiquement si None)
        
        Returns:
            conversation_id: ID unique de la conversation
        """
        if conversation_id is None:
            conversation_id = str(uuid4())
        
        # Vérifier que le template existe pour ce modèle
        try:
            TemplateRegistry.get_template(model_name)
        except Exception:
            # Si le template n'existe pas, on le laisse échouer proprement
            pass
        
        context = ConversationContext(
            conversation_id=conversation_id,
            model_name=model_name,
            gpu_id=gpu_id,
            max_messages=max_messages or self.default_max_messages,
            system_prompt=system_prompt
        )
        
        # Ajouter le prompt système comme message si fourni
        if system_prompt:
            context.add_message(MessageRole.SYSTEM, system_prompt)
        
        self._active_conversations[conversation_id] = context
        
        if self.auto_save:
            self.storage.save_conversation(context)
        
        return conversation_id
    
    def send_message(
        self,
        conversation_id: str,
        user_message: str,
        temperature: float = 0.7,
        max_new_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Envoie un message et récupère la réponse.
        
        Processus :
        1. Récupère le contexte de la conversation
        2. Ajoute le message utilisateur au contexte
        3. Formate le contexte selon le template du modèle
        4. Envoie à l'API via LLMClient
        5. Ajoute la réponse au contexte
        6. Sauvegarde (si auto_save activé)
        
        Args:
            conversation_id: ID de la conversation
            user_message: Message de l'utilisateur
            temperature: Température pour la génération (0.0-2.0, défaut: 0.7)
            max_new_tokens: Nombre maximum de nouveaux tokens (1-4096, défaut: 150)
        
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        
        Raises:
            ConversationNotFoundError: Si la conversation n'existe pas
        """
        # 1. Récupérer le contexte
        context = self.get_conversation(conversation_id)
        if context is None:
            raise ConversationNotFoundError(conversation_id)
        
        # 2. Ajouter message utilisateur
        context.add_message(MessageRole.USER, user_message)
        
        # 3. Obtenir le template
        template = TemplateRegistry.get_template(context.model_name)
        
        # 4. Obtenir les messages formatés (méthode extensible pour résumé futur)
        messages_to_format = context.get_messages_for_formatting()
        
        # 5. Formater le contexte selon le template
        formatted_prompt = template.format_conversation(
            messages_to_format,
            context.system_prompt
        )
        
        # 6. Envoyer à l'API
        response_data = self.client.chat.send_message(
            gpu_id=context.gpu_id,
            message=formatted_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        # 7. Extraire la réponse
        assistant_response = response_data.get("response", "")
        
        # 8. Ajouter au contexte
        context.add_message(MessageRole.ASSISTANT, assistant_response)
        
        # 9. Sauvegarder
        if self.auto_save:
            self.storage.save_conversation(context)
        
        return {
            "response": assistant_response,
            "conversation_id": conversation_id,
            "model_name": context.model_name,
            "gpu_id": context.gpu_id,
            "message_count": context.get_message_count(),
            **response_data
        }
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        Récupère une conversation (active ou depuis le stockage).
        
        Args:
            conversation_id: ID de la conversation
        
        Returns:
            Le contexte de conversation ou None si introuvable
        """
        # Vérifier d'abord les conversations actives
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]
        
        # Charger depuis le stockage
        context = self.storage.load_conversation(conversation_id)
        if context:
            self._active_conversations[conversation_id] = context
        
        return context
    
    def list_conversations(
        self, 
        model_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Liste les IDs des conversations.
        
        Args:
            model_name: Filtrer par nom de modèle (optionnel)
            limit: Limiter le nombre de résultats (optionnel)
        
        Returns:
            Liste des IDs de conversations
        """
        return self.storage.list_conversations(model_name=model_name, limit=limit)
    
    def delete_conversation(self, conversation_id: str):
        """
        Supprime une conversation.
        
        Args:
            conversation_id: ID de la conversation à supprimer
        
        Raises:
            ConversationNotFoundError: Si la conversation n'existe pas
        """
        # Vérifier que la conversation existe
        if not self.get_conversation(conversation_id):
            raise ConversationNotFoundError(conversation_id)
        
        # Retirer du cache actif
        if conversation_id in self._active_conversations:
            del self._active_conversations[conversation_id]
        
        # Supprimer du stockage
        self.storage.delete_conversation(conversation_id)
    
    def clear_conversation_context(self, conversation_id: str, keep_system: bool = True):
        """
        Vide le contexte d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            keep_system: Si True, conserve les messages système
        """
        context = self.get_conversation(conversation_id)
        if context is None:
            raise ConversationNotFoundError(conversation_id)
        
        context.clear_context(keep_system=keep_system)
        
        if self.auto_save:
            self.storage.save_conversation(context)
    
    def update_conversation_max_messages(
        self, 
        conversation_id: str, 
        max_messages: Optional[int]
    ):
        """
        Met à jour le nombre maximum de messages pour une conversation.
        
        Args:
            conversation_id: ID de la conversation
            max_messages: Nouveau nombre maximum (None = pas de limite)
        """
        context = self.get_conversation(conversation_id)
        if context is None:
            raise ConversationNotFoundError(conversation_id)
        
        context.max_messages = max_messages
        context._truncate_if_needed()
        
        if self.auto_save:
            self.storage.save_conversation(context)

