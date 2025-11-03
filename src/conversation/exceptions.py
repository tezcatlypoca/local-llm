"""
Exceptions personnalisées pour le module de conversations.
"""


class ConversationError(Exception):
    """Exception de base pour les erreurs de conversation."""
    pass


class ConversationNotFoundError(ConversationError):
    """Conversation introuvable."""
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        super().__init__(f"Conversation '{conversation_id}' introuvable")


class TemplateNotFoundError(ConversationError):
    """Template non trouvé pour le modèle."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Aucun template trouvé pour le modèle '{model_name}'")


class ContextTooLargeError(ConversationError):
    """Le contexte dépasse la limite autorisée."""
    
    def __init__(self, current_size: int, max_size: int):
        self.current_size = current_size
        self.max_size = max_size
        super().__init__(
            f"Le contexte dépasse la limite autorisée: {current_size} > {max_size}"
        )

