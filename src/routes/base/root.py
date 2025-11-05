"""
Route root de l'API.
"""
from flask import Blueprint, jsonify
from ...core.api_client import base_api_client
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('root', __name__)


@bp.route('/', methods=['GET'])
def root():
    """
    Route de base pour vérifier que l'API est active.
    
    GET /
    """
    try:
        result = base_api_client.root.check_status()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API de base: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la communication avec l'API de base: {str(e)}"
        }), 500

