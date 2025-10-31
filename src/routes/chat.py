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

logger = logging.getLogger(__name__)

# Création du Blueprint
chat_bp = Blueprint('chat', __name__)


def _validate_chat_messages(messages):
    """
    Valide et prépare les messages pour /chat.
    API pure : on accepte les messages tels quels sans ajouter de formatage ou contexte.
    
    Formats acceptés:
    - list de strings : ["message1", "message2", ...]
    - list de dicts avec 'content' : [{"content": "..."}, ...]
    - list de dicts avec 'role' et 'content' : [{"role": "...", "content": "..."}]
    
    L'API ne fait que concaténer les messages, le formatage/contextualisation est géré par l'app client.
    
    Returns:
        (is_valid: bool, error_message: str, formatted_prompt: str)
    """
    if messages is None:
        return False, "Le champ 'messages' est requis pour /chat.", None
    
    # Format list requis
    if not isinstance(messages, list):
        return False, "Le champ 'messages' doit être une liste.", None
    
    if len(messages) == 0:
        return False, "La liste de messages ne peut pas être vide.", None
    
    # Extraire le contenu de chaque message (format flexible)
    prompt_parts = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, str):
            # Message simple (string)
            if not msg.strip():
                return False, f"Le message à l'index {idx} ne peut pas être vide.", None
            prompt_parts.append(msg.strip())
        elif isinstance(msg, dict):
            # Message dict - extraire le contenu (on ignore les rôles, formatage géré par l'app)
            content = msg.get("content") or msg.get("text") or msg.get("message")
            if content is None:
                return False, f"Le message à l'index {idx} doit contenir 'content', 'text' ou 'message'.", None
            if not isinstance(content, str) or not content.strip():
                return False, f"Le contenu du message à l'index {idx} ne peut pas être vide.", None
            prompt_parts.append(content.strip())
        else:
            return False, f"Le message à l'index {idx} doit être une string ou un dictionnaire.", None
    
    # Concaténation simple avec des sauts de ligne (sans formatage de rôles)
    formatted_prompt = "\n".join(prompt_parts)
    
    return True, None, formatted_prompt


@chat_bp.route('/chat/<int:gpu_id>', methods=['POST'])
def chat(gpu_id: int):
    """
    Génère une réponse de chat conversationnel en utilisant le modèle chargé sur le GPU spécifié.
    
    Args:
        gpu_id: Numéro du GPU (0 ou 1) contenant le modèle à utiliser
    
    Body JSON requis:
        {
            "messages": list,         # Requis: liste de messages (strings ou dicts avec 'content')
                                     # Exemples: ["msg1", "msg2"] ou [{"content": "msg1"}, ...]
                                     # L'API concatène simplement les messages, le formatage est géré par l'app client
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
        
        # Extraire et valider les messages (format conversationnel)
        messages = request_data.get('messages')
        is_valid, error_msg, formatted_message = _validate_chat_messages(messages)
        
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400
        
        # Extraire les paramètres optionnels avec valeurs par défaut
        temperature = request_data.get('temperature', 0.7)
        max_new_tokens = request_data.get('max_new_tokens', 512)
        
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
        
        # Vérifier que le gestionnaire a un modèle chargé sur ce GPU
        manager = get_llm_manager()
        gpu_status = manager.get_model_status(gpu_id=gpu_id)
        
        if "error" in gpu_status:
            return jsonify({
                'status': 'error',
                'message': gpu_status["error"]
            }), 400
        
        if not gpu_status.get("gpu_available", False):
            return jsonify({
                'status': 'error',
                'message': f'GPU {gpu_id} n\'est pas disponible.'
            }), 404
        
        if not gpu_status.get("model_loaded", False):
            return jsonify({
                'status': 'error',
                'message': f'Aucun modèle n\'est chargé sur le GPU {gpu_id}. Chargez un modèle avec POST /models/load/{{model_name}} d\'abord.'
            }), 404
        
        model_name = gpu_status.get("model_name", "unknown")
        
        # Générer la réponse
        logger.info(f"Génération de réponse sur GPU {gpu_id} avec modèle {model_name}...")
        response = manager.generate(
            prompt=formatted_message,
            gpu_id=gpu_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2  # Pénalité contre les répétitions (augmentée pour chat)
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
        temperature = request_data.get('temperature', 0.7)
        max_new_tokens = request_data.get('max_new_tokens', 512)
        
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
        
        # Vérifier que le gestionnaire a un modèle chargé sur ce GPU
        manager = get_llm_manager()
        gpu_status = manager.get_model_status(gpu_id=gpu_id)
        
        if "error" in gpu_status:
            return jsonify({
                'status': 'error',
                'message': gpu_status["error"]
            }), 400
        
        if not gpu_status.get("gpu_available", False):
            return jsonify({
                'status': 'error',
                'message': f'GPU {gpu_id} n\'est pas disponible.'
            }), 404
        
        if not gpu_status.get("model_loaded", False):
            return jsonify({
                'status': 'error',
                'message': f'Aucun modèle n\'est chargé sur le GPU {gpu_id}. Chargez un modèle avec POST /models/load/{{model_name}} d\'abord.'
            }), 404
        
        model_name = gpu_status.get("model_name", "unknown")
        
        # Générer la réponse
        logger.info(f"Génération de completion sur GPU {gpu_id} avec modèle {model_name}...")
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

