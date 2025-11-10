"""
Tests unitaires pour les routes Flask.
"""

import pytest
from flask import Flask
from src.main import app
from src.db.database import Database
import tempfile
import os


@pytest.fixture
def client():
    """Fixture pour créer un client de test Flask."""
    # Utiliser une base de données temporaire pour les tests
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    app.config['TESTING'] = True
    app.config['DATABASE'] = db_path
    
    with app.test_client() as client:
        yield client
    
    # Nettoyer
    os.close(db_fd)
    if os.path.exists(db_path):
        os.unlink(db_path)


class TestConversationsRoutes:
    """Tests pour les routes de conversations."""
    
    def test_get_conversations_empty(self, client):
        """Test GET /conversations avec aucune conversation."""
        response = client.get('/conversations')
        if response.status_code != 200:
            # Afficher l'erreur pour debug
            error_data = response.get_json()
            print(f"\n=== ERREUR DEBUG ===")
            print(f"Status: {response.status_code}")
            print(f"Error data: {error_data}")
            print(f"===================\n")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}. Error: {response.get_json()}"
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_create_conversation_valid(self, client):
        """Test POST /conversations avec données valides."""
        response = client.post(
            '/conversations',
            json={
                "model_name": "mistral",
                "provider": "local",
                "name": "Test Conversation"
            }
        )
        assert response.status_code == 201
        data = response.get_json()
        assert data['model_name'] == "mistral"
        assert data['provider'] == "local"
        assert data['name'] == "Test Conversation"
        assert 'id' in data
    
    def test_create_conversation_missing_model_name(self, client):
        """Test POST /conversations sans model_name."""
        response = client.post('/conversations', json={})
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_create_conversation_invalid_provider(self, client):
        """Test POST /conversations avec provider invalide."""
        response = client.post(
            '/conversations',
            json={
                "model_name": "mistral",
                "provider": "invalid"
            }
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_get_conversation_not_found(self, client):
        """Test GET /conversations/{id} avec ID inexistant."""
        response = client.get('/conversations/999')
        assert response.status_code == 404


class TestMessagesRoutes:
    """Tests pour les routes de messages."""
    
    def test_post_message_no_conversation(self, client):
        """Test POST /conversations/{id}/message avec conversation inexistante."""
        response = client.post(
            '/conversations/999/message',
            json={"content": "Hello"}
        )
        assert response.status_code == 404
    
    def test_post_message_missing_content(self, client):
        """Test POST /conversations/{id}/message sans content."""
        # Créer d'abord une conversation
        conv_response = client.post(
            '/conversations',
            json={"model_name": "mistral"}
        )
        conv_id = conv_response.get_json()['id']
        
        # Tenter d'envoyer un message sans content
        response = client.post(
            f'/conversations/{conv_id}/message',
            json={}
        )
        assert response.status_code == 400
    
    def test_post_message_empty_content(self, client):
        """Test POST /conversations/{id}/message avec content vide."""
        # Créer d'abord une conversation
        conv_response = client.post(
            '/conversations',
            json={"model_name": "mistral"}
        )
        conv_id = conv_response.get_json()['id']
        
        # Tenter d'envoyer un message avec content vide
        response = client.post(
            f'/conversations/{conv_id}/message',
            json={"content": ""}
        )
        assert response.status_code == 400

