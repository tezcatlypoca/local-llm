import sys
from pathlib import Path
import os
import logging

# Ajouter la racine du projet au PYTHONPATH si nécessaire
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Supprimer l'avertissement de transformers concernant PyTorch/TensorFlow
# (on n'a besoin que des tokenizers, pas des modèles complets)
logging.getLogger("transformers").setLevel(logging.ERROR)

from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.clients.base_api.base_api_client import BaseApiClient
from src.routes.conversations_route import conversations_bp
from src.routes.messages_route import messages_bp
from src.routes.providers_route import providers_bp

app = Flask(__name__)
base_api_client = BaseApiClient()

# Configuration CORS
# Permet les requêtes depuis n'importe quelle origine (à restreindre en production)
CORS(app, resources={
    r"/*": {
        "origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("RATELIMIT_STORAGE_URL", "memory://")
)

# Enregistrer les Blueprints
app.register_blueprint(conversations_bp)
app.register_blueprint(messages_bp)
app.register_blueprint(providers_bp)

# Initialiser les rate limiters dans les blueprints APRÈS l'enregistrement
from src.routes.conversations_route import init_limiter as init_conversations_limiter
from src.routes.messages_route import init_limiter as init_messages_limiter
init_conversations_limiter(limiter)
init_messages_limiter(limiter)

@app.route('/', methods=['GET'])
def root():
    base_api_response = base_api_client.root()
    is_base_api_connected = base_api_response is not None
    
    if is_base_api_connected:
        base_api_status = 200
        base_api_message = 'Base API connected'
    else:
        base_api_status = 503
        base_api_message = 'Base API not connected'
    
    return jsonify(
        {
            'base-api': 
            {
                'message': base_api_message,
                'status': base_api_status
            },
            'overlay-api': 
                {
                    'message': 'Overlay API ON',
                    'status': 200
                }
        }
    )

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
    "swagger": "2.0",
    "info": {
        "title": "LLM Inference Overlay API",
        "description": "API de surcouche pour gérer les conversations, messages, contextes et inférences LLM",
        "version": "1.0.0",
        "contact": {
            "name": "API Support"
        }
    },
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {
            "name": "Conversations",
            "description": "Gestion des conversations"
        },
        {
            "name": "Messages",
            "description": "Gestion des messages dans les conversations"
        },
        {
            "name": "Providers",
            "description": "Gestion des providers et modèles disponibles"
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)