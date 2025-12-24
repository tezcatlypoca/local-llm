"""
Configuration partagée pour les routes RAG.

Ce module contient :
- L'instance globale du RAGManager
- Les configurations communes (upload folder, extensions autorisées)
- La fonction d'initialisation du RAGManager
"""

import os
import logging
import tempfile
from typing import Optional

from src.rag import RAGManager

logger = logging.getLogger(__name__)

# Instance globale du RAGManager
rag_manager: Optional[RAGManager] = None

# Configuration pour l'upload de fichiers
ALLOWED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx'}
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'rag_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def init_rag_manager(
    collection_name: str = "rag_collection",
    persist_directory: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    device: Optional[str] = None
):
    """
    Initialise le RAGManager global.
    
    Args:
        collection_name: Nom de la collection ChromaDB
        persist_directory: Répertoire pour persister la base de données
        embedding_model: Nom du modèle d'embedding
        chunk_size: Taille des chunks
        chunk_overlap: Chevauchement entre chunks
        device: Device pour les embeddings
    """
    global rag_manager
    try:
        rag_manager = RAGManager(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            device=device
        )
        logger.info("RAGManager initialisé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du RAGManager: {e}")
        raise


def get_rag_manager() -> Optional[RAGManager]:
    """Retourne l'instance du RAGManager."""
    return rag_manager


def allowed_file(filename: str) -> bool:
    """Vérifie si le fichier a une extension autorisée."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

