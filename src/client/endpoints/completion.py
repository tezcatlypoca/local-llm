"""
Endpoint pour les completions de texte avec les modèles.
"""
from typing import Dict, Any, Optional
from .base import BaseEndpoint


class CompletionEndpoint(BaseEndpoint):
    """
    Endpoint pour générer des completions de texte simple en utilisant les modèles chargés.
    """
    
    def complete(
        self,
        gpu_id: int,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Génère une completion de texte simple en utilisant le modèle chargé sur le GPU.
        
        Args:
            gpu_id: Numéro du GPU (0 ou 1) contenant le modèle à utiliser
            prompt: Prompt texte simple à compléter
            temperature: Température pour la génération (0.0-2.0, défaut: 0.7)
            max_new_tokens: Nombre maximum de nouveaux tokens à générer (1-4096, défaut: 150)
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "success",
                "response": str,  # Texte complété généré par le modèle
                "gpu_id": int,
                "model_name": str,
                "parameters": {
                    "temperature": float,
                    "max_new_tokens": int
                }
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur 
                               (400 si prompt invalide, 404 si GPU non disponible 
                               ou modèle non chargé)
            requests.RequestException: Si la requête échoue
        """
        path = f"/completion/{gpu_id}"
        json_data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }
        
        return self.post(path, json=json_data)

