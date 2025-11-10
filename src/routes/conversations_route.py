from src.services.conversation_manager import ConversationManager
from src.utils.validators import CreateConversationRequest
from flask import Blueprint, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

# Créer un Blueprint pour les routes de conversations
conversations_bp = Blueprint('conversations', __name__)
conversation_manager = ConversationManager()

# Rate limiter - sera initialisé dans main.py
limiter = None

def init_limiter(app_limiter):
    """Initialise le rate limiter pour ce blueprint et applique les limites."""
    global limiter
    limiter = app_limiter
    # Appliquer les limites après initialisation
    if limiter is not None:
        # Appliquer le rate limiting à la fonction create_conversation
        create_conversation._rate_limit_applied = True
        limiter.limit("10 per minute")(create_conversation)

@conversations_bp.route("/conversations", methods=["GET"])
def get_conversations():
    try:
        conversations = conversation_manager.get_conversations()
        return jsonify([conv.to_dict() for conv in conversations]), 200
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des conversations: {e}", exc_info=True)
        return jsonify(error=str(e)), 500

@conversations_bp.route("/conversations/<int:id>", methods=["GET"])
def get_conversation(id: int):
    try:
        conversation = conversation_manager.get_conversation(id)
        if conversation is not None:
            return jsonify(conversation.to_dict()), 200
        else:
            return jsonify(error=f"Conversation {id} not found"), 404
    except Exception as e:
        return jsonify(error=str(e)), 500
    
@conversations_bp.route("/conversations", methods=["DELETE"])
def delete_all_conversations():
    """
    Supprime toutes les conversations.
    ---
    tags:
      - Conversations
    responses:
      200:
        description: Toutes les conversations ont été supprimées avec succès
        schema:
          type: object
          properties:
            message:
              type: string
              example: "5 conversation(s) deleted"
            count:
              type: integer
              example: 5
      500:
        description: Erreur serveur
    """
    try:
        count = conversation_manager.delete_all_conversations()
        return jsonify(message=f"{count} conversation(s) deleted", count=count), 200
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de toutes les conversations: {e}", exc_info=True)
        return jsonify(error=str(e)), 500

@conversations_bp.route("/conversations/<int:id>", methods=["DELETE"])
def delete_conversation(id: int):
    """
    Supprime une conversation spécifique.
    ---
    tags:
      - Conversations
    parameters:
      - in: path
        name: id
        type: integer
        required: true
        description: ID de la conversation à supprimer
    responses:
      200:
        description: Conversation supprimée avec succès
      404:
        description: Conversation non trouvée
      500:
        description: Erreur serveur
    """
    try: 
        deleted = conversation_manager.delete_conversation(id)
        if deleted:
            return jsonify(message=f"Conversation {id} deleted"), 200
        else:
            return jsonify(error=f"Conversation {id} not found"), 404
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la conversation {id}: {e}", exc_info=True)
        return jsonify(error=str(e)), 500
    
@conversations_bp.route("/conversations", methods=["POST"])
def create_conversation():
    """
    Crée une nouvelle conversation.
    ---
    tags:
      - Conversations
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - model_name
          properties:
            model_name:
              type: string
              description: Nom du modèle à utiliser
              example: "mistral"
            provider:
              type: string
              enum: [local, groq]
              description: Provider à utiliser (optionnel, défaut: local)
              example: "local"
            name:
              type: string
              description: Nom de la conversation (optionnel)
              example: "Ma conversation"
            temperature:
              type: number
              minimum: 0.0
              maximum: 2.0
              description: Température pour la génération (optionnel, défaut: 0.7)
              example: 0.7
            message_max:
              type: integer
              minimum: 0
              maximum: 1000
              description: Nombre maximum de messages (optionnel, défaut: 10)
              example: 10
            system_prompt:
              type: string
              maxLength: 2000
              description: Message système pour guider le modèle (optionnel, un prompt par défaut sera utilisé si non fourni)
              example: "Tu es un assistant utile. Réponds toujours en français."
    responses:
      201:
        description: Conversation créée avec succès
      400:
        description: Erreur de validation
      500:
        description: Erreur serveur
    """
    try:
        # Validation avec Pydantic
        data = request.get_json()
        if not data:
            return jsonify(error="Request body is required"), 400
        
        try:
            validated_data = CreateConversationRequest(**data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(f"{field}: {error['msg']}")
            return jsonify(error="Validation error", details=errors), 400
        
        # Créer la conversation avec les données validées
        conversation = conversation_manager.create_conversation(
            model_name=validated_data.model_name,
            provider=validated_data.provider,
            name=validated_data.name,
            temperature=validated_data.temperature,
            message_max=validated_data.message_max,
            system_prompt=validated_data.system_prompt
        )
        return jsonify(conversation.to_dict()), 201
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        logger.error(f"Erreur lors de la création de la conversation: {e}", exc_info=True)
        return jsonify(error="Internal server error"), 500
