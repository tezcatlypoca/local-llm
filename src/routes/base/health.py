"""
Routes pour le health check.
"""
from flask import Blueprint, jsonify
from ...core.api_client import base_api_client
import requests
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('health', __name__)


@bp.route('/health', methods=['GET'])
def health():
    """
    Retourne l'état de santé global de l'API et de tous les GPUs.
    
    GET /health
    """
    try:
        result = base_api_client.health.check_health()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors du health check global: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la communication avec l'API de base: {str(e)}"
        }), 500


@bp.route('/health/<int:gpu_id>', methods=['GET'])
def health_gpu(gpu_id: int):
    """
    Retourne l'état de santé détaillé d'un GPU spécifique.
    
    GET /health/<gpu_id>
    """
    try:
        result = base_api_client.health.check_gpu_health(gpu_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors du health check GPU {gpu_id}: {e}")
        # Essayer de récupérer le code d'erreur HTTP si disponible
        status_code = 500
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
        
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la communication avec l'API de base: {str(e)}"
        }), status_code

