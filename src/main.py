"""
Application Flask principale pour l'API Local LLM avec module RAG générique.
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

from src.routes.rag import rag_bp, init_rag_manager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Enregistrer les blueprints
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
    Route root pour vérifier que l'API est active.
    
    Returns:
        JSON: Message de statut de l'API
    """
    return jsonify({
        "message": "API Flask active",
        "status": "ok"
    }), 200


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

