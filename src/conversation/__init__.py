"""
Module de gestion des conversations avec contexte, templates et persistance.
"""
from .manager import ConversationManager
from .context import ConversationContext, Message, MessageRole
from .exceptions import (
    ConversationError,
    ConversationNotFoundError,
    TemplateNotFoundError,
    ContextTooLargeError
)

__all__ = [
    "ConversationManager",
    "ConversationContext",
    "Message",
    "MessageRole",
    "ConversationError",
    "ConversationNotFoundError",
    "TemplateNotFoundError",
    "ContextTooLargeError",
]

