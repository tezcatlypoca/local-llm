"""
Configuration pytest pour les tests.
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH pour les imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from flask import Flask
from src.main import app
from src.db.database import Database
from src.repositories.conversation_repository import ConversationRepository
from src.services.conversation_manager import ConversationManager
import tempfile
import os


@pytest.fixture
def client():
    """Fixture pour créer un client de test Flask avec une base de données temporaire."""
    # Utiliser une base de données temporaire pour les tests
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    
    # Réinitialiser le pool global pour permettre un nouveau pool pour les tests
    from src.db.database import Database
    with Database._pool_lock:
        if Database._pool is not None:
            Database._pool.close_all()
        Database._pool = None
    
    # Créer une instance de Database avec le chemin temporaire
    test_db = Database(db_path=db_path)
    
    # Créer un repository avec cette base de données
    test_repository = ConversationRepository(database=test_db)
    
    # Créer un ConversationManager avec ce repository
    test_manager = ConversationManager(repository=test_repository)
    
    # Remplacer le conversation_manager dans le blueprint
    from src.routes import conversations_route
    original_manager = conversations_route.conversation_manager
    conversations_route.conversation_manager = test_manager
    
    # Faire de même pour messages_route
    from src.routes import messages_route
    original_messages_manager = messages_route.conversation_manager
    messages_route.conversation_manager = test_manager
    
    app.config['TESTING'] = True
    app.config['DATABASE'] = db_path
    
    with app.test_client() as client:
        yield client
    
    # Restaurer les managers originaux
    conversations_route.conversation_manager = original_manager
    messages_route.conversation_manager = original_messages_manager
    
    # Fermer la base de données de test
    test_db.close()
    
    # Réinitialiser le pool pour le prochain test
    with Database._pool_lock:
        if Database._pool is not None:
            Database._pool.close_all()
        Database._pool = None
    
    # Nettoyer
    os.close(db_fd)
    if os.path.exists(db_path):
        os.unlink(db_path)

