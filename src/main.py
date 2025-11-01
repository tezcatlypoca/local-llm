import logging
from flask import Flask, jsonify
from flask_cors import CORS
from routes.models import models_bp
from routes.model_management import model_management_bp
from routes.health import health_bp
from routes.chat import chat_bp
from routes.logs import logs_bp
from utils.log_buffer import setup_log_buffer_handler

app = Flask(__name__)

# Configuration CORS pour permettre les requêtes depuis les fichiers HTML locaux et autres origines
# Cela permet notamment d'utiliser le viewer HTML depuis le système de fichiers
CORS(app, resources={
    r"/logs/*": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"]
    },
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"]
    }
})

# Configuration du logging avec buffer en mémoire
# Cela capture tous les logs pour le streaming en temps réel via l'API
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ajouter le handler de buffer pour capturer tous les logs
setup_log_buffer_handler()  # Root logger
setup_log_buffer_handler('src')  # Logger pour src/
setup_log_buffer_handler('routes')  # Logger pour routes/
setup_log_buffer_handler('llm_manager')  # Logger pour llm_manager

logger = logging.getLogger(__name__)
logger.info("Système de logging avec buffer initialisé")

# Enregistrement des Blueprints
app.register_blueprint(models_bp)
app.register_blueprint(model_management_bp)
app.register_blueprint(health_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(logs_bp)


@app.route('/', methods=['GET'])
def root():
    """Route root de l'API."""
    return jsonify({
        'message': 'API Flask active',
        'status': 'ok'
    })


@app.route('/logs/viewer', methods=['GET'])
def logs_viewer():
    """
    Route pour servir le viewer HTML des logs.
    Permet d'éviter les problèmes CORS en servant le fichier depuis Flask.
    """
    from flask import send_from_directory
    import os
    
    # Chemin vers le fichier HTML
    doc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'doc')
    return send_from_directory(doc_dir, 'logs_viewer_example.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

