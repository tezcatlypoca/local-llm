"""
Routes de health check et informations système pour le RAG.
"""

import logging
from flask import Blueprint, jsonify

from src.rag import DocumentLoader
from src.routes.rag.config import get_rag_manager

logger = logging.getLogger(__name__)

# Créer le blueprint pour les routes de health
health_bp = Blueprint('rag_health', __name__)


@health_bp.route('/health', methods=['GET'])
def rag_health():
    """
    Vérifie l'état du système RAG.
    
    Returns:
        JSON avec le statut du système et les informations de la collection
    """
    try:
        rag_manager = get_rag_manager()
        if rag_manager is None:
            return jsonify({
                "status": "error",
                "message": "RAGManager non initialisé"
            }), 503
        
        info = rag_manager.get_collection_info()
        return jsonify({
            "status": "ok",
            "rag_system": "active",
            "collection_info": info
        }), 200
    except Exception as e:
        logger.error(f"Erreur lors du health check RAG: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@health_bp.route('/formats', methods=['GET'])
def get_supported_formats():
    """
    Retourne la liste des formats de fichiers supportés.
    
    Returns:
        JSON avec la liste des formats supportés
    """
    try:
        formats = DocumentLoader.get_supported_formats()
        return jsonify({
            "status": "success",
            "supported_formats": formats,
            "count": len(formats)
        }), 200
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des formats: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

