"""
Routes pour la completion de texte.
"""
from flask import Blueprint, jsonify, request
from ...core.api_client import base_api_client
import requests
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('completion', __name__)


@bp.route('/completion/<int:gpu_id>', methods=['POST'])
def completion(gpu_id: int):
    """
    Génère une completion de texte simple en utilisant le modèle chargé sur le GPU spécifié.
    
    POST /completion/<gpu_id>
    
    Body JSON requis:
    {
        "prompt": "Le machine learning est une branche de",
        "temperature": 0.7,  # optionnel
        "max_new_tokens": 150  # optionnel
    }
    """
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({
                "status": "error",
                "message": "Le champ 'prompt' doit être une chaîne de caractères (str)."
            }), 400
        
        if 'prompt' not in json_data:
            return jsonify({
                "status": "error",
                "message": "Le champ 'prompt' doit être une chaîne de caractères (str)."
            }), 400
        
        if not isinstance(json_data['prompt'], str):
            return jsonify({
                "status": "error",
                "message": "Le champ 'prompt' doit être une chaîne de caractères (str)."
            }), 400
        
        result = base_api_client.completion.complete(
            gpu_id=gpu_id,
            prompt=json_data['prompt'],
            temperature=json_data.get('temperature', 0.7),
            max_new_tokens=json_data.get('max_new_tokens', 150)
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la completion sur GPU {gpu_id}: {e}")
        # Essayer de récupérer le code d'erreur HTTP si disponible
        status_code = 500
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
        
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la génération: {str(e)}"
        }), status_code

