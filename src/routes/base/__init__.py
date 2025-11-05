"""
Routes de l'API de base (gestion des modèles, GPUs, chat, logs, etc.).
"""
from .root import bp as root_bp
from .models import bp as models_bp
from .health import bp as health_bp
from .chat import bp as chat_bp
from .completion import bp as completion_bp
from .logs import bp as logs_bp

__all__ = [
    'root_bp',
    'models_bp',
    'health_bp',
    'chat_bp',
    'completion_bp',
    'logs_bp'
]
