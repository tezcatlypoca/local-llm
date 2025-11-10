from src.clients.providers.provider_factory import ProviderFactory
from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)

# Créer un Blueprint pour les routes de providers
providers_bp = Blueprint('providers', __name__)


@providers_bp.route("/providers/<provider_name>/models", methods=["GET"])
def get_provider_models(provider_name: str):
    """
    Récupère la liste des modèles disponibles pour un provider.
    ---
    tags:
      - Providers
    parameters:
      - in: path
        name: provider_name
        type: string
        enum: [local, groq]
        required: true
        description: Nom du provider (local ou groq)
    responses:
      200:
        description: Liste des modèles disponibles
        schema:
          type: object
          properties:
            provider:
              type: string
              example: "local"
            models:
              type: array
              items:
                type: object
                properties:
                  identifier:
                    type: string
                    example: "mistral"
      400:
        description: Provider invalide
      500:
        description: Erreur serveur
    """
    try:
        # Normaliser le nom du provider
        provider_name = provider_name.lower()
        
        # Valider le provider
        if provider_name not in ['local', 'groq']:
            return jsonify(error=f"Provider '{provider_name}' non reconnu. Utilisez 'local' ou 'groq'"), 400
        
        # Récupérer le provider
        try:
            provider = ProviderFactory.get_provider(provider_name)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        
        # Vérifier si le provider est disponible
        if not provider.is_available():
            return jsonify(
                error=f"Provider '{provider_name}' n'est pas disponible",
                provider=provider_name
            ), 503
        
        # Récupérer les modèles disponibles
        try:
            models = provider.get_available_models()
            return jsonify(
                provider=provider_name,
                models=models,
                count=len(models)
            ), 200
        except ConnectionError as e:
            logger.error(f"Erreur de connexion avec le provider {provider_name}: {e}")
            return jsonify(
                error=f"Impossible de récupérer les modèles du provider '{provider_name}'",
                details=str(e),
                provider=provider_name
            ), 503
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles: {e}", exc_info=True)
            return jsonify(
                error=f"Erreur lors de la récupération des modèles",
                details=str(e),
                provider=provider_name
            ), 500
            
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}", exc_info=True)
        return jsonify(error="Internal server error"), 500


@providers_bp.route("/providers", methods=["GET"])
def get_providers():
    """
    Liste tous les providers disponibles.
    ---
    tags:
      - Providers
    responses:
      200:
        description: Liste des providers disponibles
        schema:
          type: object
          properties:
            providers:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                    example: "local"
                  available:
                    type: boolean
                    example: true
    """
    try:
        providers_list = []
        available_providers = ProviderFactory.get_available_providers()
        
        for provider_name in available_providers:
            try:
                provider = ProviderFactory.get_provider(provider_name)
                is_available = provider.is_available()
                providers_list.append({
                    "name": provider_name,
                    "available": is_available
                })
            except Exception as e:
                logger.warning(f"Impossible de vérifier la disponibilité du provider '{provider_name}': {e}")
                providers_list.append({
                    "name": provider_name,
                    "available": False
                })
        
        return jsonify(providers=providers_list, count=len(providers_list)), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des providers: {e}", exc_info=True)
        return jsonify(error="Internal server error"), 500

