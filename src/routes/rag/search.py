"""
Routes de recherche et récupération de contexte pour le RAG.
"""

import logging
from flask import Blueprint, request, jsonify

from src.routes.rag.config import get_rag_manager

logger = logging.getLogger(__name__)

# Créer le blueprint pour les routes de recherche
search_bp = Blueprint('rag_search', __name__)


@search_bp.route('/search', methods=['POST'])
def search_documents():
    """
    Recherche des documents pertinents.
    
    Body JSON:
    {
        "query": "Votre question ici",
        "n_results": 5,  // optionnel, défaut: 5
        "filter_metadata": {}  // optionnel
    }
    
    Returns:
        JSON avec les résultats de recherche
    """
    rag_manager = get_rag_manager()
    if rag_manager is None:
        return jsonify({
            "status": "error",
            "message": "RAGManager non initialisé"
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Body JSON requis"
            }), 400
        
        query = data.get('query')
        if not query or not isinstance(query, str):
            return jsonify({
                "status": "error",
                "message": "Le champ 'query' (string) est requis"
            }), 400
        
        n_results = data.get('n_results', 5)
        if not isinstance(n_results, int) or n_results < 1:
            return jsonify({
                "status": "error",
                "message": "Le champ 'n_results' doit être un entier positif"
            }), 400
        
        filter_metadata = data.get('filter_metadata')
        
        results = rag_manager.search(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
        
        return jsonify({
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": results
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la recherche: {str(e)}"
        }), 500


@search_bp.route('/context', methods=['POST'])
def get_context():
    """
    Récupère le contexte formaté pour une requête.
    
    Body JSON:
    {
        "query": "Votre question ici",
        "n_results": 5,  // optionnel
        "filter_metadata": {}  // optionnel
    }
    
    Returns:
        JSON avec le contexte formaté
    """
    rag_manager = get_rag_manager()
    if rag_manager is None:
        return jsonify({
            "status": "error",
            "message": "RAGManager non initialisé"
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Body JSON requis"
            }), 400
        
        query = data.get('query')
        if not query or not isinstance(query, str):
            return jsonify({
                "status": "error",
                "message": "Le champ 'query' (string) est requis"
            }), 400
        
        n_results = data.get('n_results', 5)
        filter_metadata = data.get('filter_metadata')
        
        context = rag_manager.get_context_for_query(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
        
        return jsonify({
            "status": "success",
            "query": query,
            "context": context,
            "context_length": len(context)
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du contexte: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la récupération du contexte: {str(e)}"
        }), 500

