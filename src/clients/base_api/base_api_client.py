import requests, os
from typing import Optional

from src.clients.base_api.endpoints.models import ModelEndpoints


class BaseApiClient:


    def __init__(self):
        self.base_url = os.getenv('BASE_API_URL', 'localhost:8000')
        self.models = ModelEndpoints(self)
    
    @property
    def get_base_url(self) -> str:
        return self.base_url
    
    def root(self):
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()  # Lève une exception si le status code est >= 400
            return response.json()
        except (requests.exceptions.RequestException, requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return None
    
    def health(self, gpu_id: Optional[int] = None):
        if gpu_id is not None:
            response = requests.get(f"{self.base_url}/health/{gpu_id}").json()
            # print(response)
        else:
            response = requests.get(f"{self.base_url}/health").json()
            # print(response)
        return response