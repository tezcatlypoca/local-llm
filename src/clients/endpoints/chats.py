import requests

class ChatEndpoints:

    def __init__(self, base_client):
        self.base_url = f"{base_client.get_base_url}/chat"
        self.base_client = base_client
    
    def post_chat(self, message):
        # message doit etre un json object comprenant le template adapté au model avec lequel on interagit
        response = requests.post(f"{self.base_url}", message).json()
        return response
    