"""
Système de tracking des modèles chargés sur les GPUs.
Permet de garder en mémoire les gpu_id et access_token pour chaque modèle chargé.
"""
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Informations sur un modèle chargé."""
    model_name: str
    gpu_id: int
    access_token: str
    loaded_at: datetime


class ModelTracker:
    """
    Tracker pour suivre les modèles chargés sur les GPUs.
    
    Permet de stocker les gpu_id et access_token pour faciliter
    la gestion des modèles depuis la surcouche.
    """
    
    def __init__(self):
        """Initialise le tracker."""
        # Dict: model_name -> LoadedModel
        self._loaded_models: Dict[str, LoadedModel] = {}
        # Dict: gpu_id -> model_name (pour vérifier rapidement)
        self._gpu_to_model: Dict[int, str] = {}
    
    def register_loaded_model(
        self,
        model_name: str,
        gpu_id: int,
        access_token: str
    ):
        """
        Enregistre un modèle chargé.
        
        Args:
            model_name: Nom du modèle
            gpu_id: ID du GPU sur lequel le modèle est chargé
            access_token: Token d'accès pour décharger le modèle
        """
        # Si le GPU avait déjà un modèle, on le remplace
        if gpu_id in self._gpu_to_model:
            old_model_name = self._gpu_to_model[gpu_id]
            if old_model_name in self._loaded_models:
                del self._loaded_models[old_model_name]
        
        loaded_model = LoadedModel(
            model_name=model_name,
            gpu_id=gpu_id,
            access_token=access_token,
            loaded_at=datetime.now()
        )
        
        self._loaded_models[model_name] = loaded_model
        self._gpu_to_model[gpu_id] = model_name
        
        logger.info(f"Modèle '{model_name}' enregistré sur GPU {gpu_id}")
    
    def unregister_model(self, gpu_id: int):
        """
        Désenregistre un modèle déchargé.
        
        Args:
            gpu_id: ID du GPU duquel le modèle a été déchargé
        """
        if gpu_id in self._gpu_to_model:
            model_name = self._gpu_to_model[gpu_id]
            if model_name in self._loaded_models:
                del self._loaded_models[model_name]
            del self._gpu_to_model[gpu_id]
            logger.info(f"Modèle '{model_name}' désenregistré de GPU {gpu_id}")
    
    def get_model_info(self, model_name: str) -> Optional[LoadedModel]:
        """
        Récupère les informations d'un modèle chargé.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            LoadedModel ou None si le modèle n'est pas chargé
        """
        return self._loaded_models.get(model_name)
    
    def get_gpu_info(self, gpu_id: int) -> Optional[LoadedModel]:
        """
        Récupère les informations du modèle chargé sur un GPU.
        
        Args:
            gpu_id: ID du GPU
            
        Returns:
            LoadedModel ou None si aucun modèle n'est chargé sur ce GPU
        """
        if gpu_id not in self._gpu_to_model:
            return None
        
        model_name = self._gpu_to_model[gpu_id]
        return self._loaded_models.get(model_name)
    
    def get_access_token(self, gpu_id: int) -> Optional[str]:
        """
        Récupère le token d'accès pour un GPU.
        
        Args:
            gpu_id: ID du GPU
            
        Returns:
            Access token ou None
        """
        model_info = self.get_gpu_info(gpu_id)
        return model_info.access_token if model_info else None
    
    def list_loaded_models(self) -> Dict[str, LoadedModel]:
        """Retourne tous les modèles chargés."""
        return self._loaded_models.copy()


# Instance globale
model_tracker = ModelTracker()

