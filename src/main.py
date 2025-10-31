from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    """Route root de l'API."""
    return jsonify({
        'message': 'API Flask active',
        'status': 'ok'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

