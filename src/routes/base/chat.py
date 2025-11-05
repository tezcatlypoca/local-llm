"""
Routes pour le chat (génération de texte).
"""
from flask import Blueprint, jsonify, request
from ...core.api_client import base_api_client
import requests
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('chat', __name__)


@bp.route('/chat/<int:gpu_id>', methods=['POST'])
def chat(gpu_id: int):
    """
    Génère une réponse en utilisant le modèle chargé sur le GPU spécifié.
    
    POST /chat/<gpu_id>
    
    Body JSON requis:
    {
        "message": "Votre message ici" ou {"text": "..."} ou {"content": "..."},
        "temperature": 0.7,  # optionnel
        "max_new_tokens": 150  # optionnel
    }
    
    Note: Le champ "prompt" est aussi accepté comme alternative à "message".
    """
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({
                "status": "error",
                "message": "Le body JSON doit contenir soit un champ 'message' (str ou objet) soit un champ 'prompt' (str)."
            }), 400
        
        # Vérifier qu'on a soit 'message' soit 'prompt'
        if 'message' not in json_data and 'prompt' not in json_data:
            return jsonify({
                "status": "error",
                "message": "Le body JSON doit contenir soit un champ 'message' (str ou objet) soit un champ 'prompt' (str)."
            }), 400
        
        # Utiliser send_message ou send_prompt selon ce qui est fourni
        if 'message' in json_data:
            result = base_api_client.chat.send_message(
                gpu_id=gpu_id,
                message=json_data['message'],
                temperature=json_data.get('temperature', 0.7),
                max_new_tokens=json_data.get('max_new_tokens', 150)
            )
        elif 'prompt' in json_data:
            result = base_api_client.chat.send_prompt(
                gpu_id=gpu_id,
                prompt=json_data['prompt'],
                temperature=json_data.get('temperature', 0.7),
                max_new_tokens=json_data.get('max_new_tokens', 150)
            )
        else:
            # Ne devrait pas arriver (déjà validé avant)
            result = {}
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la génération de chat sur GPU {gpu_id}: {e}")
        # Essayer de récupérer le code d'erreur HTTP si disponible
        status_code = 500
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
        
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la génération: {str(e)}"
        }), status_code

