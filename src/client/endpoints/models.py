"""
Endpoint pour la gestion des modèles (liste, chargement, déchargement).
"""
from typing import Dict, Any, List, Optional
from .base import BaseEndpoint


class ModelsEndpoint(BaseEndpoint):
    """
    Endpoint pour interagir avec les modèles de l'API.
    
    Permet de lister les modèles disponibles, charger un modèle sur un GPU,
    et décharger un modèle d'un GPU.
    """
    
    def list_models(self) -> Dict[str, Any]:
        """
        Liste tous les modèles LLM disponibles localement.
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "success",
                "count": int,
                "cache_dir": str,
                "cache_exists": bool,
                "models": List[Dict]  # Liste des modèles avec leurs informations
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur
            requests.RequestException: Si la requête échoue
        """
        return self.get("/models")
    
    def load_model(
        self, 
        model_name: str, 
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Charge un modèle LLM sur un GPU libre.
        
        Args:
            model_name: Nom/identifiant du modèle Hugging Face 
                       (ex: "gpt2", "microsoft/phi-2")
            model_kwargs: Arguments optionnels pour le chargement du modèle
                         (ex: {"torch_dtype": "float16"})
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "success",
                "message": str,
                "gpu_id": int,
                "gpu_identifier": str,
                "model_name": str,
                "access_token": str,  # ⚠️ IMPORTANT: À conserver pour décharger le modèle
                "note": str,
                "gpu_status": Dict
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur (404 si modèle introuvable, 
                               503 si aucun GPU libre)
            requests.RequestException: Si la requête échoue
        """
        path = f"/models/load/{model_name}"
        json_data = {}
        if model_kwargs:
            json_data["model_kwargs"] = model_kwargs
        
        return self.post(path, json=json_data if json_data else None)
    
    def unload_model(self, gpu_id: int, access_token: str) -> Dict[str, Any]:
        """
        Décharge un modèle d'un GPU spécifique.
        
        Args:
            gpu_id: Numéro du GPU (0 ou 1)
            access_token: Token d'accès reçu lors du chargement du modèle
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "success",
                "message": str,
                "gpu_id": int,
                "gpu_identifier": str
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur 
                               (400 si token manquant, 403 si token invalide, 
                               404 si aucun modèle sur le GPU)
            requests.RequestException: Si la requête échoue
        """
        path = f"/models/unload/{gpu_id}"
        json_data = {"access_token": access_token}
        
        return self.post(path, json=json_data)

