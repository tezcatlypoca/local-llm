"""
Endpoint pour les vérifications de santé (health checks).
"""
from typing import Dict, Any, Optional
from .base import BaseEndpoint


class HealthEndpoint(BaseEndpoint):
    """
    Endpoint pour les vérifications de santé de l'API et des GPUs.
    
    Permet de vérifier l'état global de l'API et l'état détaillé des GPUs.
    """
    
    def check_health(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé global de l'API et de tous les GPUs.
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "healthy",
                "service": "local-llm-api",
                "total_gpus": int,
                "gpus": {
                    "0": {...},  # Informations détaillées sur GPU 0
                    "1": {...}   # Informations détaillées sur GPU 1
                },
                "summary": {
                    "total_gpus_available": int,
                    "gpus_with_models": int,
                    "gpus_free": int
                }
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur
            requests.RequestException: Si la requête échoue
        """
        return self.get("/health")
    
    def check_gpu_health(self, gpu_id: int) -> Dict[str, Any]:
        """
        Vérifie l'état de santé détaillé d'un GPU spécifique.
        
        Args:
            gpu_id: Numéro du GPU (0 ou 1)
        
        Returns:
            Dictionnaire contenant:
            {
                "status": str,  # "free", "in_use", "unavailable"
                "gpu_id": int,
                "gpu_identifier": str,
                "available": bool,
                "model": {
                    "loaded": bool,
                    "name": Optional[str],
                    "has_access_token": bool
                },
                "device": str,
                "message": str,
                "metrics": {
                    "memory": {...},
                    "gpu_info": {...}
                }
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur (400 si GPU ID invalide)
            requests.RequestException: Si la requête échoue
        """
        path = f"/health/{gpu_id}"
        return self.get(path)

