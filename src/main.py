from flask import Flask, jsonify
from clients.base_api_client import BaseApiClient

app = Flask(__name__)
base_api_client = BaseApiClient()

@app.route('/', methods=['GET'])
def root():
    base_api_response = base_api_client.root()
    is_base_api_connected = base_api_response is not None
    
    if is_base_api_connected:
        base_api_status = 200
        base_api_message = 'Base API connected'
    else:
        base_api_status = 503
        base_api_message = 'Base API not connected'
    
    return jsonify(
        {
            'base-api': 
            {
                'message': base_api_message,
                'status': base_api_status
            },
            'overlay-api': 
                {
                    'message': 'Overlay API ON',
                    'status': 200
                }
        }
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)