"""
Initialisation du ConversationManager pour l'API surcouche.
"""
import logging
from pathlib import Path
from ..client import LLMClient
from ..conversation import ConversationManager
from ..conversation.storage.json_storage import JSONStorage
from .api_client import get_base_api_client

logger = logging.getLogger(__name__)


def get_conversation_manager() -> ConversationManager:
    """
    Crée et retourne une instance de ConversationManager configurée.
    
    Utilise le LLMClient configuré pour appeler l'API de base.
    Utilise JSONStorage pour persister les conversations dans src/data/.
    """
    # Récupérer le client API de base
    client = get_base_api_client()
    
    # Créer le stockage JSON (facilement modifiable pour passer à SQLite)
    # Pour utiliser SQLite, remplacer par:
    # from ..conversation.storage.sqlite_storage import SQLiteStorage
    # storage = SQLiteStorage(db_path="src/data/conversations.db")
    storage = JSONStorage(data_dir="src/data")
    
    # Créer le gestionnaire de conversations avec auto-save activé
    manager = ConversationManager(
        client=client,
        storage=storage,  # Utilise JSONStorage au lieu de SQLite par défaut
        auto_save=True,  # Sauvegarde automatique après chaque message
        default_max_messages=50  # Limite par défaut de 50 messages
    )
    
    logger.info("ConversationManager initialisé avec JSONStorage")
    return manager


# Instance globale du gestionnaire
conversation_manager = get_conversation_manager()

