import requests

class ModelEndpoints:


    def __init__(self, base_client):
        self.base_client = base_client
        self.base_url = f"{self.base_client.get_base_url}/models"
    
    def get_models(self):
        response = requests.get(f"{self.base_url}").json()
        # print(response)
        return response
    
    def get_models_gguf(self):
        response = requests.get(f"{self.base_url}/gguf").json()
        # print(response)
        return response
    
    def post_load(self, path:str, model_name:str):
        response = requests.post(f'{self.base_url}/load/{path}/{model_name}').json()
        return response
    
    def post_unload(self, gpu_id: int):
        response = requests.post(f"{self.base_url}/unload/{gpu_id}").json()
        return response