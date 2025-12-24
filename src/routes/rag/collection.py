"""
Routes de gestion de la collection RAG.
"""

import logging
from flask import Blueprint, jsonify

from src.routes.rag.config import get_rag_manager

logger = logging.getLogger(__name__)

# Créer le blueprint pour les routes de collection
collection_bp = Blueprint('rag_collection', __name__)


@collection_bp.route('/collection/info', methods=['GET'])
def get_collection_info():
    """
    Retourne des informations sur la collection.
    
    Returns:
        JSON avec les informations de la collection
    """
    rag_manager = get_rag_manager()
    if rag_manager is None:
        return jsonify({
            "status": "error",
            "message": "RAGManager non initialisé"
        }), 503
    
    try:
        info = rag_manager.get_collection_info()
        return jsonify({
            "status": "success",
            "collection_info": info
        }), 200
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@collection_bp.route('/collection/delete', methods=['DELETE'])
def delete_collection():
    """
    Supprime la collection actuelle.
    
    Returns:
        JSON avec le statut de la suppression
    """
    rag_manager = get_rag_manager()
    if rag_manager is None:
        return jsonify({
            "status": "error",
            "message": "RAGManager non initialisé"
        }), 503
    
    try:
        rag_manager.delete_collection()
        return jsonify({
            "status": "success",
            "message": "Collection supprimée avec succès"
        }), 200
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la collection: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

