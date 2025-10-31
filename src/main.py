from flask import Flask, jsonify
from routes.models import models_bp
from routes.model_management import model_management_bp
from routes.health import health_bp
from routes.chat import chat_bp

app = Flask(__name__)

# Enregistrement des Blueprints
app.register_blueprint(models_bp)
app.register_blueprint(model_management_bp)
app.register_blueprint(health_bp)
app.register_blueprint(chat_bp)


@app.route('/', methods=['GET'])
def root():
    """Route root de l'API."""
    return jsonify({
        'message': 'API Flask active',
        'status': 'ok'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

