"""
Module RAG - Routes organisées par thème.

Ce module regroupe toutes les routes RAG dans des fichiers séparés par thème :
- health.py : Health check et formats
- documents.py : Ajout et upload de documents
- search.py : Recherche et contexte
- collection.py : Gestion de la collection
"""

from flask import Blueprint

from src.routes.rag.config import init_rag_manager
from src.routes.rag import health, documents, search, collection

# Créer le blueprint principal
rag_bp = Blueprint('rag', __name__, url_prefix='/rag')

# Enregistrer les sous-blueprints
rag_bp.register_blueprint(health.health_bp)
rag_bp.register_blueprint(documents.documents_bp)
rag_bp.register_blueprint(search.search_bp)
rag_bp.register_blueprint(collection.collection_bp)

# Exporter la fonction d'initialisation
__all__ = ['rag_bp', 'init_rag_manager']

