"""
API Flask surcouche pour les briques API.
Cette API fait des appels proxy vers l'API de base existante.
"""
from flask import Flask, jsonify
from flask_cors import CORS
from src.routes.base import (
    root_bp,
    models_bp,
    health_bp,
    chat_bp,
    completion_bp,
    logs_bp
)
from src.routes.conversations import bp as conversations_bp

app = Flask(__name__)

# Activer CORS pour permettre les appels depuis différents domaines
CORS(app)

# Enregistrer les blueprints sans préfixe pour transparence avec l'API de base
app.register_blueprint(root_bp)
app.register_blueprint(models_bp)
app.register_blueprint(health_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(completion_bp)
app.register_blueprint(logs_bp)

# Routes spécifiques à la surcouche (conversations)
app.register_blueprint(conversations_bp)


# La route root est gérée par le blueprint root_bp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

