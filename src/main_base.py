"""
Application Flask pour l'API de base (inférence LLM locale).
Version sans rate limiting pour usage interne.
"""

import sys
from pathlib import Path
import os
import logging
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Ajouter la racine du projet au PYTHONPATH si nécessaire
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supprimer l'avertissement de transformers concernant PyTorch/TensorFlow
logging.getLogger("transformers").setLevel(logging.ERROR)

from flask import Flask, jsonify
from flask_cors import CORS

# Imports des clients et routes
from src.clients.base_api.base_api_client import BaseApiClient
from src.routes.conversations_route import conversations_bp
from src.routes.messages_route import messages_bp
from src.routes.providers_route import providers_bp
from src.routes.rag import rag_bp, init_rag_manager

app = Flask(__name__)
base_api_client = BaseApiClient()

# Configuration CORS
CORS(app, resources={
    r"/*": {
        "origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# PAS de Rate Limiting pour l'API de base (usage interne)

# Enregistrer les Blueprints
app.register_blueprint(conversations_bp)
app.register_blueprint(messages_bp)
app.register_blueprint(providers_bp)
app.register_blueprint(rag_bp)

# Initialiser le RAGManager
try:
    # Configuration depuis les variables d'environnement ou valeurs par défaut
    rag_collection_name = os.getenv('RAG_COLLECTION_NAME', 'rag_collection')
    rag_persist_dir = os.getenv('RAG_PERSIST_DIR', './data/rag_db')
    rag_embedding_model = os.getenv('RAG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    rag_chunk_size = int(os.getenv('RAG_CHUNK_SIZE', '1000'))
    rag_chunk_overlap = int(os.getenv('RAG_CHUNK_OVERLAP', '200'))
    rag_device = os.getenv('RAG_DEVICE', None)  # None = auto
    
    init_rag_manager(
        collection_name=rag_collection_name,
        persist_directory=rag_persist_dir,
        embedding_model=rag_embedding_model,
        chunk_size=rag_chunk_size,
        chunk_overlap=rag_chunk_overlap,
        device=rag_device
    )
    logger.info("RAGManager initialisé avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation du RAGManager: {e}")
    logger.warning("L'API fonctionnera mais les routes RAG ne seront pas disponibles")


@app.route('/', methods=['GET'])
def root():
    """
    Route root pour vérifier que l'API de base est active.
    
    Returns:
        JSON: Statut de l'API de base
    """
    return jsonify({
        'base-api': {
            'message': 'Base API ON',
            'status': 200
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Gestionnaire d'erreur 404."""
    return jsonify({
        "status": "error",
        "message": "Route non trouvée"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestionnaire d'erreur 500."""
    logger.error(f"Erreur interne: {error}")
    return jsonify({
        "status": "error",
        "message": "Erreur interne du serveur"
    }), 500


# Configuration Swagger/OpenAPI
from flasgger import Swagger

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api-docs"
}

swagger_template = {
    "info": {
        "title": "Local LLM Base API",
        "description": "API de base pour l'inférence LLM locale",
        "version": "3.0.0",
        "contact": {
            "name": "API Support"
        }
    },
    "schemes": ["http", "https"],
    "tags": [
        {
            "name": "Conversations",
            "description": "Gestion des conversations"
        },
        {
            "name": "Messages",
            "description": "Gestion des messages"
        },
        {
            "name": "Providers",
            "description": "Gestion des providers LLM"
        },
        {
            "name": "RAG",
            "description": "Système de Retrieval-Augmented Generation"
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
