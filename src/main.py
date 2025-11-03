"""
API Flask de base pour les briques API.
"""
from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    """Route racine de l'API."""
    return jsonify({
        'message': 'API Flask active',
        'status': 'ok'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

