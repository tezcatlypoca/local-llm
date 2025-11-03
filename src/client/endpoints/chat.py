"""
Endpoint pour les conversations de chat avec les modèles.
"""
from typing import Dict, Any, Union, Optional
from .base import BaseEndpoint


class ChatEndpoint(BaseEndpoint):
    """
    Endpoint pour générer des réponses de chat en utilisant les modèles chargés.
    
    ⚠️ IMPORTANT: Cette route accepte un message simple sans formatage de 
    contexte/template. La gestion du contexte et du formatage doit être effectuée 
    par la surcouche API appelante.
    """
    
    def send_message(
        self,
        gpu_id: int,
        message: Union[str, Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Envoie un message et génère une réponse en utilisant le modèle chargé sur le GPU.
        
        Args:
            gpu_id: Numéro du GPU (0 ou 1) contenant le modèle à utiliser
            message: Message texte simple
                - Format string: "Votre message ici"
                - Format dict: {"text": "..."} ou {"content": "..."}
            temperature: Température pour la génération (0.0-2.0, défaut: 0.7)
            max_new_tokens: Nombre maximum de nouveaux tokens à générer (1-4096, défaut: 150)
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "success",
                "response": str,  # Réponse générée par le modèle
                "gpu_id": int,
                "model_name": str,
                "parameters": {
                    "temperature": float,
                    "max_new_tokens": int
                }
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur 
                               (400 si paramètres invalides, 404 si GPU non disponible 
                               ou modèle non chargé)
            requests.RequestException: Si la requête échoue
        """
        path = f"/chat/{gpu_id}"
        json_data = {
            "message": message,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }
        
        return self.post(path, json=json_data)
    
    def send_prompt(
        self,
        gpu_id: int,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Envoie un prompt (alternative à send_message, utilise le champ "prompt").
        
        Args:
            gpu_id: Numéro du GPU (0 ou 1) contenant le modèle à utiliser
            prompt: Prompt texte simple
            temperature: Température pour la génération (0.0-2.0, défaut: 0.7)
            max_new_tokens: Nombre maximum de nouveaux tokens à générer (1-4096, défaut: 150)
        
        Returns:
            Dictionnaire contenant la réponse (identique à send_message)
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur
            requests.RequestException: Si la requête échoue
        """
        path = f"/chat/{gpu_id}"
        json_data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }
        
        return self.post(path, json=json_data)

