"""
Routes pour l'inférence de chat avec les modèles LLM.
"""
import logging
import sys
from pathlib import Path
from flask import Blueprint, jsonify, request

# Import depuis le dossier parent (src/)
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from llm_manager_instance import get_llm_manager
from llm_gguf_manager_instance import get_gguf_manager

logger = logging.getLogger(__name__)

# Création du Blueprint
chat_bp = Blueprint('chat', __name__)


def _extract_message(request_data):
    """
    Extrait le message de la requête. Accepte plusieurs formats.
    
    Formats acceptés:
    - "message": str - Message simple (prioritaire)
    - "prompt": str - Alternative pour compatibilité
    - Objet avec champ "text" ou "content"
    
    Returns:
        (success: bool, error_message: str, message_text: str or None)
    """
    # Priorité 1: champ "message"
    if "message" in request_data:
        message = request_data["message"]
        if isinstance(message, str):
            if not message.strip():
                return False, "Le champ 'message' ne peut pas être vide.", None
            return True, None, message.strip()
        elif isinstance(message, dict):
            # Objet avec champ text ou content
            text = message.get("text") or message.get("content") or message.get("message")
            if text and isinstance(text, str) and text.strip():
                return True, None, text.strip()
            return False, "Le champ 'message' (objet) doit contenir 'text', 'content' ou 'message' avec une valeur non vide.", None
        else:
            return False, "Le champ 'message' doit être une chaîne de caractères (str) ou un objet avec 'text'/'content'.", None
    
    # Priorité 2: champ "prompt" (compatibilité)
    if "prompt" in request_data:
        prompt = request_data["prompt"]
        if isinstance(prompt, str):
            if not prompt.strip():
                return False, "Le champ 'prompt' ne peut pas être vide.", None
            return True, None, prompt.strip()
        else:
            return False, "Le champ 'prompt' doit être une chaîne de caractères (str).", None
    
    return False, "Le body JSON doit contenir soit un champ 'message' (str ou objet) soit un champ 'prompt' (str).", None


@chat_bp.route('/chat/<int:gpu_id>', methods=['POST'])
def chat(gpu_id: int):
    """
    Génère une réponse en utilisant le modèle chargé sur le GPU spécifié.
    
    Cette route accepte un message simple sans formatage de contexte/template.
    La gestion du contexte et du formatage doit être effectuée par la surcouche API appelante.
    
    Args:
        gpu_id: Numéro du GPU (0 ou 1) contenant le modèle à utiliser
    
    Body JSON requis:
        {
            "message": str ou objet,  # Requis: message texte simple
                                     # Format str: "Votre message ici"
                                     # Format objet: {"text": "..."} ou {"content": "..."}
            "temperature": float,     # Optionnel: température pour la génération (défaut: 0.7)
            "max_new_tokens": int     # Optionnel: nombre max de nouveaux tokens (défaut: 512)
        }
    
    Alternative (compatibilité):
        {
            "prompt": str,            # Alternative au champ "message"
            ...
        }
    
    Returns:
        - 200: Réponse générée avec succès
        - 400: Erreur de validation (paramètres manquants ou invalides)
        - 404: GPU non disponible ou modèle non chargé
        - 500: Erreur lors de la génération
    """
    try:
        # Vérifier que le GPU ID est valide
        if gpu_id not in [0, 1]:
            return jsonify({
                'status': 'error',
                'message': f'GPU ID invalide: {gpu_id}. Doit être 0 ou 1.'
            }), 400
        
        # Récupérer les données du body JSON (force=True permet d'accepter même sans Content-Type)
        try:
            request_data = request.get_json(force=True, silent=True)
        except Exception:
            request_data = None
        
        if request_data is None:
            return jsonify({
                'status': 'error',
                'message': 'Le body de la requête doit être au format JSON. Assurez-vous d\'envoyer du JSON et de définir le header Content-Type: application/json dans Postman.'
            }), 400
        
        # Extraire le message (format simple, sans formatage)
        success, error_msg, message_text = _extract_message(request_data)
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400
        
        # Extraire les paramètres optionnels avec valeurs par défaut
        # Valeurs par défaut optimisées : température modérée pour éviter les problèmes numériques,
        # max_new_tokens réduit pour éviter les réponses trop longues/hors sujet
        # Note: température minimale recommandée: 0.5 pour éviter les NaN/inf dans les probabilités
        temperature = request_data.get('temperature', 0.7)
        max_new_tokens = request_data.get('max_new_tokens', 150)
        
        # Valider les types des paramètres optionnels
        if not isinstance(temperature, (int, float)):
            return jsonify({
                'status': 'error',
                'message': f'Le paramètre "temperature" doit être un nombre (float), reçu: {type(temperature).__name__}'
            }), 400
        
        if not isinstance(max_new_tokens, int):
            return jsonify({
                'status': 'error',
                'message': f'Le paramètre "max_new_tokens" doit être un entier (int), reçu: {type(max_new_tokens).__name__}'
            }), 400
        
        # Valider les plages de valeurs
        if not (0.0 <= temperature <= 2.0):
            return jsonify({
                'status': 'error',
                'message': 'La température doit être entre 0.0 et 2.0.'
            }), 400
        
        if max_new_tokens <= 0:
            return jsonify({
                'status': 'error',
                'message': 'max_new_tokens doit être un entier positif.'
            }), 400
        
        if max_new_tokens > 4096:
            return jsonify({
                'status': 'error',
                'message': 'max_new_tokens ne peut pas dépasser 4096.'
            }), 400
        
        # Détecter le type de modèle chargé (transformers ou GGUF)
        transformers_manager = get_llm_manager()
        gguf_manager = get_gguf_manager()
        
        transformers_status = transformers_manager.get_model_status(gpu_id=gpu_id)
        gguf_status = gguf_manager.get_model_status(gpu_id=gpu_id)
        
        # Utiliser le gestionnaire qui a un modèle chargé
        if transformers_status.get("model_loaded", False):
            manager = transformers_manager
            gpu_status = transformers_status
            model_type = "transformers"
        elif gguf_status.get("model_loaded", False):
            manager = gguf_manager
            gpu_status = gguf_status
            model_type = "gguf"
        else:
            return jsonify({
                'status': 'error',
                'message': f'Aucun modèle n\'est chargé sur le GPU {gpu_id}. Chargez un modèle avec POST /models/load/{{model_name}} d\'abord.'
            }), 404
        
        if "error" in gpu_status:
            return jsonify({
                'status': 'error',
                'message': gpu_status["error"]
            }), 400
        
        if model_type == "transformers" and not gpu_status.get("gpu_available", False):
            return jsonify({
                'status': 'error',
                'message': f'GPU {gpu_id} n\'est pas disponible.'
            }), 404
        
        model_name = gpu_status.get("model_name", "unknown")
        
        # Générer la réponse (message passé tel quel, sans formatage)
        logger.info(f"Génération de réponse sur GPU {gpu_id} avec modèle {model_name} (type: {model_type})...")
        
        if model_type == "gguf":
            # Pour GGUF, pas de do_sample, utiliser les paramètres directement
            response = manager.generate(
                prompt=message_text,
                gpu_id=gpu_id,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=0.9,
                repetition_penalty=1.2
            )
        else:
            # Pour transformers, utiliser do_sample
            response = manager.generate(
                prompt=message_text,
                gpu_id=gpu_id,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        if response is None:
            return jsonify({
                'status': 'error',
                'message': 'Erreur lors de la génération de la réponse.'
            }), 500
        
        # Retourner la réponse au format JSON structuré
        return jsonify({
            'status': 'success',
            'response': response,
            'gpu_id': gpu_id,
            'model_name': model_name,
            'parameters': {
                'temperature': temperature,
                'max_new_tokens': max_new_tokens
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de chat: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur serveur: {str(e)}'
        }), 500


@chat_bp.route('/completion/<int:gpu_id>', methods=['POST'])
def completion(gpu_id: int):
    """
    Génère une completion de texte simple en utilisant le modèle chargé sur le GPU spécifié.
    
    Args:
        gpu_id: Numéro du GPU (0 ou 1) contenant le modèle à utiliser
    
    Body JSON requis:
        {
            "prompt": str,            # Requis: prompt texte simple
            "temperature": float,     # Optionnel: température pour la génération (défaut: 0.7)
            "max_new_tokens": int     # Optionnel: nombre max de nouveaux tokens (défaut: 512)
        }
    
    Returns:
        - 200: Réponse générée avec succès
        - 400: Erreur de validation (paramètres manquants ou invalides)
        - 404: GPU non disponible ou modèle non chargé
        - 500: Erreur lors de la génération
    """
    try:
        # Vérifier que le GPU ID est valide
        if gpu_id not in [0, 1]:
            return jsonify({
                'status': 'error',
                'message': f'GPU ID invalide: {gpu_id}. Doit être 0 ou 1.'
            }), 400
        
        # Récupérer les données du body JSON (force=True permet d'accepter même sans Content-Type)
        try:
            request_data = request.get_json(force=True, silent=True)
        except Exception:
            request_data = None
        
        if request_data is None:
            return jsonify({
                'status': 'error',
                'message': 'Le body de la requête doit être au format JSON. Assurez-vous d\'envoyer du JSON et de définir le header Content-Type: application/json dans Postman.'
            }), 400
        
        # Extraire et valider le prompt (format string simple)
        prompt = request_data.get('prompt')
        
        if prompt is None:
            return jsonify({
                'status': 'error',
                'message': 'Le champ "prompt" est requis pour /completion.'
            }), 400
        
        if not isinstance(prompt, str):
            return jsonify({
                'status': 'error',
                'message': 'Le champ "prompt" doit être une chaîne de caractères (str).'
            }), 400
        
        if not prompt.strip():
            return jsonify({
                'status': 'error',
                'message': 'Le prompt ne peut pas être vide.'
            }), 400
        
        # Extraire les paramètres optionnels avec valeurs par défaut
        # Valeurs par défaut optimisées : température modérée pour éviter les problèmes numériques,
        # max_new_tokens réduit pour éviter les réponses trop longues/hors sujet
        # Note: température minimale recommandée: 0.5 pour éviter les NaN/inf dans les probabilités
        temperature = request_data.get('temperature', 0.7)
        max_new_tokens = request_data.get('max_new_tokens', 150)
        
        # Valider les types des paramètres optionnels
        if not isinstance(temperature, (int, float)):
            return jsonify({
                'status': 'error',
                'message': f'Le paramètre "temperature" doit être un nombre (float), reçu: {type(temperature).__name__}'
            }), 400
        
        if not isinstance(max_new_tokens, int):
            return jsonify({
                'status': 'error',
                'message': f'Le paramètre "max_new_tokens" doit être un entier (int), reçu: {type(max_new_tokens).__name__}'
            }), 400
        
        # Valider les plages de valeurs
        if not (0.0 <= temperature <= 2.0):
            return jsonify({
                'status': 'error',
                'message': 'La température doit être entre 0.0 et 2.0.'
            }), 400
        
        if max_new_tokens <= 0:
            return jsonify({
                'status': 'error',
                'message': 'max_new_tokens doit être un entier positif.'
            }), 400
        
        if max_new_tokens > 4096:
            return jsonify({
                'status': 'error',
                'message': 'max_new_tokens ne peut pas dépasser 4096.'
            }), 400
        
        # Détecter le type de modèle chargé (transformers ou GGUF)
        transformers_manager = get_llm_manager()
        gguf_manager = get_gguf_manager()
        
        transformers_status = transformers_manager.get_model_status(gpu_id=gpu_id)
        gguf_status = gguf_manager.get_model_status(gpu_id=gpu_id)
        
        # Utiliser le gestionnaire qui a un modèle chargé
        if transformers_status.get("model_loaded", False):
            manager = transformers_manager
            gpu_status = transformers_status
            model_type = "transformers"
        elif gguf_status.get("model_loaded", False):
            manager = gguf_manager
            gpu_status = gguf_status
            model_type = "gguf"
        else:
            return jsonify({
                'status': 'error',
                'message': f'Aucun modèle n\'est chargé sur le GPU {gpu_id}. Chargez un modèle avec POST /models/load/{{model_name}} d\'abord.'
            }), 404
        
        if "error" in gpu_status:
            return jsonify({
                'status': 'error',
                'message': gpu_status["error"]
            }), 400
        
        if model_type == "transformers" and not gpu_status.get("gpu_available", False):
            return jsonify({
                'status': 'error',
                'message': f'GPU {gpu_id} n\'est pas disponible.'
            }), 404
        
        model_name = gpu_status.get("model_name", "unknown")
        
        # Générer la réponse
        logger.info(f"Génération de completion sur GPU {gpu_id} avec modèle {model_name} (type: {model_type})...")
        
        if model_type == "gguf":
            # Pour GGUF, pas de do_sample
            response = manager.generate(
                prompt=prompt.strip(),
                gpu_id=gpu_id,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=0.9
            )
        else:
            # Pour transformers, utiliser do_sample
            response = manager.generate(
                prompt=prompt.strip(),
                gpu_id=gpu_id,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9
            )
        
        if response is None:
            return jsonify({
                'status': 'error',
                'message': 'Erreur lors de la génération de la réponse.'
            }), 500
        
        # Retourner la réponse au format JSON structuré
        return jsonify({
            'status': 'success',
            'response': response,
            'gpu_id': gpu_id,
            'model_name': model_name,
            'parameters': {
                'temperature': temperature,
                'max_new_tokens': max_new_tokens
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de completion: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur serveur: {str(e)}'
        }), 500

