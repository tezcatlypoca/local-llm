from src.clients.base_api_client import BaseApiClient
from typing import List

class ModelRegistry:
    _models = {}
    _aliases = {}  # nom_court -> nom_complet
    api_client = BaseApiClient()
    
    @classmethod
    def refresh_from_api(cls):
        models = cls.api_client.models.get_models_gguf()
        
        for model in models:
            cls._models[model['name']] = model
            cls._aliases[model['alias']] = model['name']
    
    @classmethod
    def get_full_name(cls, alias: str) -> str:
        return cls._aliases.get(alias)
    
    @classmethod
    def get_full_aliases(cls) -> List[str]:
        return list(cls._aliases.keys())