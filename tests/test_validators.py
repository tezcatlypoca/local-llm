"""
Tests unitaires pour les validateurs Pydantic.
"""

import pytest
from pydantic import ValidationError
from src.utils.validators import CreateConversationRequest, PostMessageRequest


class TestCreateConversationRequest:
    """Tests pour CreateConversationRequest."""
    
    def test_valid_request(self):
        """Test avec une requête valide."""
        data = {
            "model_name": "mistral",
            "provider": "local",
            "name": "Test Conversation",
            "temperature": 0.7,
            "message_max": 10
        }
        request = CreateConversationRequest(**data)
        assert request.model_name == "mistral"
        assert request.provider == "local"
        assert request.name == "Test Conversation"
        assert request.temperature == 0.7
        assert request.message_max == 10
    
    def test_minimal_request(self):
        """Test avec seulement le champ requis."""
        data = {"model_name": "mistral"}
        request = CreateConversationRequest(**data)
        assert request.model_name == "mistral"
        assert request.provider is None
        assert request.name == "Conversation"
        assert request.temperature == 0.7
        assert request.message_max == 10
    
    def test_invalid_model_name_empty(self):
        """Test avec model_name vide."""
        with pytest.raises(ValidationError) as exc_info:
            CreateConversationRequest(model_name="")
        # Le validator personnalisé devrait lever une erreur
        assert "ne peut pas être vide" in str(exc_info.value)
    
    def test_invalid_provider(self):
        """Test avec provider invalide."""
        with pytest.raises(ValidationError) as exc_info:
            CreateConversationRequest(model_name="mistral", provider="invalid")
        # Le validator personnalisé devrait lever une erreur
        assert "doit être 'local' ou 'groq'" in str(exc_info.value)
    
    def test_invalid_temperature_too_high(self):
        """Test avec température trop élevée."""
        with pytest.raises(ValidationError) as exc_info:
            CreateConversationRequest(model_name="mistral", temperature=3.0)
        # Pydantic valide le Field constraint
        assert "less than or equal to 2" in str(exc_info.value) or "less than or equal to 2.0" in str(exc_info.value)
    
    def test_invalid_temperature_negative(self):
        """Test avec température négative."""
        with pytest.raises(ValidationError) as exc_info:
            CreateConversationRequest(model_name="mistral", temperature=-1.0)
        # Pydantic valide le Field constraint
        assert "greater than or equal to 0" in str(exc_info.value) or "greater than or equal to 0.0" in str(exc_info.value)
    
    def test_invalid_message_max_negative(self):
        """Test avec message_max négatif."""
        with pytest.raises(ValidationError) as exc_info:
            CreateConversationRequest(model_name="mistral", message_max=-1)
        assert "greater than or equal to 0" in str(exc_info.value)


class TestPostMessageRequest:
    """Tests pour PostMessageRequest."""
    
    def test_valid_request(self):
        """Test avec une requête valide."""
        data = {"content": "Bonjour, comment ça va ?"}
        request = PostMessageRequest(**data)
        assert request.content == "Bonjour, comment ça va ?"
    
    def test_invalid_content_empty(self):
        """Test avec content vide."""
        with pytest.raises(ValidationError) as exc_info:
            PostMessageRequest(content="")
        # Le validator personnalisé devrait lever une erreur
        assert "ne peut pas être vide" in str(exc_info.value)
    
    def test_invalid_content_whitespace_only(self):
        """Test avec content contenant seulement des espaces."""
        with pytest.raises(ValidationError) as exc_info:
            PostMessageRequest(content="   ")
        # Le field_validator personnalisé devrait lever une erreur
        error_str = str(exc_info.value)
        assert "ne peut pas être vide" in error_str
    
    def test_content_trimmed(self):
        """Test que le content est bien trimé."""
        data = {"content": "  Bonjour  "}
        request = PostMessageRequest(**data)
        assert request.content == "Bonjour"
