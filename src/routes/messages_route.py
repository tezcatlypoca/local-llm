from src.services.conversation_manager import ConversationManager
from src.utils.validators import PostMessageRequest
from flask import Blueprint, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

messages_bp = Blueprint('messages', __name__)
conversation_manager = ConversationManager()

# Rate limiter - sera initialisé dans main.py
limiter = None

def init_limiter(app_limiter):
    """Initialise le rate limiter pour ce blueprint et applique les limites."""
    global limiter
    limiter = app_limiter
    # Appliquer les limites après initialisation
    if limiter is not None:
        # Appliquer le rate limiting à la fonction post_message
        post_message._rate_limit_applied = True
        limiter.limit("30 per minute")(post_message)

@messages_bp.route("/conversations/<int:id>/message", methods=["POST"])
def post_message(id: int):
    """
    Envoie un message dans une conversation.
    ---
    tags:
      - Messages
    parameters:
      - in: path
        name: id
        type: integer
        required: true
        description: ID de la conversation
      - in: query
        name: provider
        type: string
        enum: [local, groq]
        description: Surcharge du provider pour ce message (optionnel)
      - in: body
        name: body
        schema:
          type: object
          required:
            - content
          properties:
            content:
              type: string
              description: Contenu du message
              example: "Bonjour, comment ça va ?"
    responses:
      200:
        description: Message envoyé avec succès
      400:
        description: Erreur de validation
      404:
        description: Conversation non trouvée
      500:
        description: Erreur serveur
    """
    try:
        # Validation avec Pydantic
        data = request.get_json()
        if not data:
            return jsonify(error="Request body is required"), 400
        
        try:
            validated_data = PostMessageRequest(**data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(f"{field}: {error['msg']}")
            return jsonify(error="Validation error", details=errors), 400
        
        # Surcharge optionnelle du provider via query parameter
        provider_override = request.args.get('provider')  # ?provider=groq
        if provider_override:
            provider_override = provider_override.lower()
            if provider_override not in ['local', 'groq']:
                return jsonify(error="provider must be 'local' or 'groq'"), 400
        
        response_message = conversation_manager.post_message(
            conversation_id=id,
            content=validated_data.content,
            provider_override=provider_override
        )
        return jsonify(response_message.to_dict()), 200
    except ValueError as e:
        return jsonify(error=str(e)), 404
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message: {e}", exc_info=True)
        return jsonify(error="Internal server error"), 500
    

@messages_bp.route("/conversations/<int:id>/message", methods=["GET"])
def get_messages(id: int):
    try:
        messages = conversation_manager.get_messages(id)
        return jsonify([msg.to_dict() for msg in messages]), 200
    except Exception as e:
        return jsonify(error=str(e)), 500
    
@messages_bp.route("/conversations/<int:id>/message/<int:message_id>", methods=["GET"])
def get_message(id: int, message_id: int):
    try:
        message = conversation_manager.get_message(id, message_id)
        if message is not None:
            return jsonify(message.to_dict()), 200
        else:
            return jsonify(error=f"Message not found"), 404
    except Exception as e:
        return jsonify(error=str(e)), 500
    
@messages_bp.route("/conversations/<int:id>/message/<int:message_id>", methods=["DELETE"])
def delete_message(id: int, message_id: int):
    try:
        deleted = conversation_manager.delete_message(id, message_id)
        if deleted:
            return jsonify(message=f"Message {message_id} deleted"), 200
        else:
            return jsonify(error=f"Message {message_id} not found"), 404
    except Exception as e:
        return jsonify(error=str(e)), 500