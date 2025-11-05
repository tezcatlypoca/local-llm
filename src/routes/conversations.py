"""
Routes pour la gestion des conversations.
"""
from flask import Blueprint, jsonify, request
from ...core.conversation_manager import conversation_manager
from ...conversation.exceptions import ConversationNotFoundError, TemplateNotFoundError
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('conversations', __name__)


@bp.route('/conversations', methods=['POST'])
def create_conversation():
    """
    Crée une nouvelle conversation.
    
    POST /conversations
    
    Body JSON:
    {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "gpu_id": 0,
        "system_prompt": "Tu es un assistant utile.",  // optionnel
        "max_messages": 50,  // optionnel
        "conversation_id": "uuid-custom"  // optionnel
    }
    """
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({
                "status": "error",
                "message": "Le body JSON est requis."
            }), 400
        
        # Validation des champs requis
        if 'model_name' not in json_data:
            return jsonify({
                "status": "error",
                "message": "Le champ 'model_name' est requis."
            }), 400
        
        if 'gpu_id' not in json_data:
            return jsonify({
                "status": "error",
                "message": "Le champ 'gpu_id' est requis."
            }), 400
        
        # Créer la conversation
        conversation_id = conversation_manager.create_conversation(
            model_name=json_data['model_name'],
            gpu_id=json_data['gpu_id'],
            system_prompt=json_data.get('system_prompt'),
            max_messages=json_data.get('max_messages'),
            conversation_id=json_data.get('conversation_id')
        )
        
        # Récupérer la conversation créée pour retourner les détails
        context = conversation_manager.get_conversation(conversation_id)
        
        return jsonify({
            "status": "success",
            "conversation_id": conversation_id,
            "model_name": context.model_name,
            "gpu_id": context.gpu_id,
            "system_prompt": context.system_prompt,
            "max_messages": context.max_messages,
            "created_at": context.created_at.isoformat()
        }), 201
        
    except TemplateNotFoundError as e:
        logger.error(f"Template non trouvé: {e}")
        return jsonify({
            "status": "error",
            "message": f"Modèle non supporté: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Erreur lors de la création de la conversation: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la création de la conversation: {str(e)}"
        }), 500


@bp.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id: str):
    """
    Récupère une conversation avec son contexte complet.
    
    GET /conversations/<conversation_id>
    """
    try:
        context = conversation_manager.get_conversation(conversation_id)
        
        if context is None:
            return jsonify({
                "status": "error",
                "message": f"Conversation '{conversation_id}' introuvable."
            }), 404
        
        # Convertir les messages en format JSON
        messages = []
        for msg in context.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            })
        
        return jsonify({
            "status": "success",
            "conversation": {
                "conversation_id": context.conversation_id,
                "model_name": context.model_name,
                "gpu_id": context.gpu_id,
                "system_prompt": context.system_prompt,
                "max_messages": context.max_messages,
                "message_count": context.get_message_count(),
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
                "messages": messages
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la conversation: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la récupération de la conversation: {str(e)}"
        }), 500


@bp.route('/conversations', methods=['GET'])
def list_conversations():
    """
    Liste les conversations disponibles.
    
    GET /conversations?model_name=...&limit=...
    
    Query params:
    - model_name (optionnel): Filtrer par modèle
    - limit (optionnel): Limiter le nombre de résultats
    """
    try:
        model_name = request.args.get('model_name')
        limit = request.args.get('limit', type=int)
        
        conversation_ids = conversation_manager.list_conversations(
            model_name=model_name,
            limit=limit
        )
        
        # Récupérer les détails de chaque conversation
        conversations = []
        for conv_id in conversation_ids:
            context = conversation_manager.get_conversation(conv_id)
            if context:
                conversations.append({
                    "conversation_id": context.conversation_id,
                    "model_name": context.model_name,
                    "gpu_id": context.gpu_id,
                    "message_count": context.get_message_count(),
                    "created_at": context.created_at.isoformat(),
                    "updated_at": context.updated_at.isoformat()
                })
        
        return jsonify({
            "status": "success",
            "count": len(conversations),
            "conversations": conversations
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la liste des conversations: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la liste des conversations: {str(e)}"
        }), 500


@bp.route('/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id: str):
    """
    Supprime une conversation et tout son contexte.
    
    DELETE /conversations/<conversation_id>
    """
    try:
        conversation_manager.delete_conversation(conversation_id)
        
        return jsonify({
            "status": "success",
            "message": f"Conversation '{conversation_id}' supprimée avec succès."
        })
        
    except ConversationNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la conversation: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la suppression de la conversation: {str(e)}"
        }), 500


@bp.route('/conversations/<conversation_id>/messages', methods=['POST'])
def send_message(conversation_id: str):
    """
    Envoie un message dans une conversation et récupère la réponse.
    
    POST /conversations/<conversation_id>/messages
    
    Body JSON:
    {
        "message": "Bonjour, comment ça va ?",
        "temperature": 0.7,  // optionnel
        "max_new_tokens": 150  // optionnel
    }
    """
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({
                "status": "error",
                "message": "Le body JSON est requis."
            }), 400
        
        if 'message' not in json_data:
            return jsonify({
                "status": "error",
                "message": "Le champ 'message' est requis."
            }), 400
        
        # Envoyer le message
        result = conversation_manager.send_message(
            conversation_id=conversation_id,
            user_message=json_data['message'],
            temperature=json_data.get('temperature', 0.7),
            max_new_tokens=json_data.get('max_new_tokens', 150)
        )
        
        return jsonify({
            "status": "success",
            "response": result.get("response"),
            "conversation_id": result.get("conversation_id"),
            "message_count": result.get("message_count"),
            "parameters": {
                "temperature": json_data.get('temperature', 0.7),
                "max_new_tokens": json_data.get('max_new_tokens', 150)
            }
        })
        
    except ConversationNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de l'envoi du message: {str(e)}"
        }), 500


@bp.route('/conversations/<conversation_id>/context', methods=['DELETE'])
def clear_context(conversation_id: str):
    """
    Vide le contexte d'une conversation.
    
    DELETE /conversations/<conversation_id>/context?keep_system=true
    
    Query params:
    - keep_system (bool, défaut: true): Garder les messages système
    """
    try:
        keep_system = request.args.get('keep_system', 'true').lower() == 'true'
        
        conversation_manager.clear_conversation_context(
            conversation_id=conversation_id,
            keep_system=keep_system
        )
        
        return jsonify({
            "status": "success",
            "message": "Contexte vidé avec succès.",
            "kept_system": keep_system
        })
        
    except ConversationNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Erreur lors du vidage du contexte: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors du vidage du contexte: {str(e)}"
        }), 500


@bp.route('/conversations/<conversation_id>/max-messages', methods=['PATCH'])
def update_max_messages(conversation_id: str):
    """
    Modifie la limite de messages pour une conversation.
    
    PATCH /conversations/<conversation_id>/max-messages
    
    Body JSON:
    {
        "max_messages": 100  // ou null pour pas de limite
    }
    """
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({
                "status": "error",
                "message": "Le body JSON est requis."
            }), 400
        
        if 'max_messages' not in json_data:
            return jsonify({
                "status": "error",
                "message": "Le champ 'max_messages' est requis."
            }), 400
        
        max_messages = json_data['max_messages']
        # Valider que c'est un entier ou None
        if max_messages is not None and not isinstance(max_messages, int):
            return jsonify({
                "status": "error",
                "message": "Le champ 'max_messages' doit être un entier ou null."
            }), 400
        
        conversation_manager.update_conversation_max_messages(
            conversation_id=conversation_id,
            max_messages=max_messages
        )
        
        return jsonify({
            "status": "success",
            "max_messages": max_messages,
            "message": "Limite de messages mise à jour avec succès."
        })
        
    except ConversationNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la limite: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la mise à jour de la limite: {str(e)}"
        }), 500

