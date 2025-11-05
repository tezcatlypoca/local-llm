"""
Routes pour la gestion des modèles.
"""
from flask import Blueprint, jsonify, request
from ...core.api_client import base_api_client
from ...core.model_tracker import model_tracker
import requests
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('models', __name__)


@bp.route('/models', methods=['GET'])
def list_models():
    """
    Liste tous les modèles LLM disponibles localement.
    
    GET /models
    """
    try:
        result = base_api_client.models.list_models()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la liste des modèles: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la communication avec l'API de base: {str(e)}"
        }), 500


@bp.route('/models/load/<path:model_name>', methods=['POST'])
def load_model(model_name: str):
    """
    Charge un modèle LLM sur un GPU libre.
    
    POST /models/load/<model_name>
    
    Body JSON (optionnel):
    {
        "model_kwargs": {
            "torch_dtype": "float16"
        }
    }
    """
    try:
        json_data = request.get_json() or {}
        # Timeout plus long pour le chargement de modèles (10 minutes)
        result = base_api_client.models.load_model(
            model_name=model_name,
            model_kwargs=json_data.get("model_kwargs"),
            timeout=600
        )
        
        # Enregistrer le modèle dans le tracker
        if result.get("status") == "success":
            model_tracker.register_loaded_model(
                model_name=model_name,
                gpu_id=result.get("gpu_id"),
                access_token=result.get("access_token")
            )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
        # Essayer de récupérer le code d'erreur HTTP si disponible
        status_code = 500
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
        
        return jsonify({
            "status": "error",
            "message": f"Erreur lors du chargement du modèle: {str(e)}"
        }), status_code


@bp.route('/models/unload/<int:gpu_id>', methods=['POST'])
def unload_model(gpu_id: int):
    """
    Décharge un modèle d'un GPU spécifique.
    
    POST /models/unload/<gpu_id>
    
    Body JSON requis:
    {
        "access_token": "token_reçu_lors_du_chargement"
    }
    """
    try:
        json_data = request.get_json()
        if not json_data or 'access_token' not in json_data:
            return jsonify({
                "status": "error",
                "message": "Token d'accès requis. Fournissez le token reçu lors du chargement du modèle dans le body JSON (access_token)."
            }), 400
        
        result = base_api_client.models.unload_model(
            gpu_id=gpu_id,
            access_token=json_data['access_token']
        )
        
        # Désenregistrer le modèle du tracker
        if result.get("status") == "success":
            model_tracker.unregister_model(gpu_id)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors du déchargement du modèle sur GPU {gpu_id}: {e}")
        # Essayer de récupérer le code d'erreur HTTP si disponible
        status_code = 500
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
        
        return jsonify({
            "status": "error",
            "message": f"Erreur lors du déchargement du modèle: {str(e)}"
        }), status_code

