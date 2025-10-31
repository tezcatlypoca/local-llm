"""
Routes pour la gestion du chargement/déchargement des modèles sur GPU.
"""
import os
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request

# Import depuis le dossier parent (src/)
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from llm_manager_instance import get_llm_manager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_manager import LLMManager

logger = logging.getLogger(__name__)

# Création du Blueprint
model_management_bp = Blueprint('model_management', __name__)


def _check_model_exists(model_name: str) -> tuple[bool, str]:
    """
    Vérifie si un modèle existe localement sur la machine.
    
    Args:
        model_name: Nom/identifiant du modèle (ex: "gpt2" ou "microsoft/phi-2")
    
    Returns:
        (exists: bool, path: str) - True si le modèle existe localement et le chemin
    """
    try:
        from huggingface_hub import scan_cache_dir
        
        # Vérifier uniquement dans le cache local
        cache_info = scan_cache_dir()
        for repo in list(cache_info.repos):
            if repo.repo_id == model_name:
                if repo.revisions:
                    return True, repo.revisions[-1].snapshot_path
                return True, None
        
        # Vérifier aussi dans un dossier local si configuré
        # (même logique que dans routes/models.py)
        project_root = Path(__file__).parent.parent.parent
        local_models_dir = project_root / "models"
        
        if local_models_dir.exists():
            # Vérifier si le modèle existe dans le dossier local
            model_dir = local_models_dir / model_name.replace("/", "_")
            if model_dir.exists() and model_dir.is_dir():
                # Vérifier qu'il y a des fichiers de modèle
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.gguf')):
                            return True, str(model_dir)
        
        # Modèle non trouvé localement
        return False, None
            
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du modèle: {e}", exc_info=True)
        return False, None


def _find_free_gpu(manager) -> int:
    """
    Trouve un GPU libre pour charger un modèle.
    
    Returns:
        GPU ID (0 ou 1) ou -1 si aucun GPU n'est libre
    """
    status = manager.get_model_status()
    
    for gpu_id in [0, 1]:
        gpu_info = status["gpus"].get(gpu_id, {})
        # Vérifier que le GPU existe et qu'aucun modèle n'est chargé
        if gpu_info.get("gpu_available", False) and not gpu_info.get("model_loaded", False):
            return gpu_id
    
    return -1


@model_management_bp.route('/models/load/<path:model_name>', methods=['POST'])
def load_model(model_name: str):
    """
    Charge un modèle sur un GPU libre.
    
    Args:
        model_name: Nom/identifiant du modèle Hugging Face (ex: "gpt2" ou "microsoft/phi-2")
    
    Returns:
        - 200: Modèle chargé avec succès
        - 404: Modèle introuvable
        - 503: Aucun GPU libre
        - 500: Erreur lors du chargement
    """
    try:
        manager = get_llm_manager()
        
        # Vérifier que le modèle existe
        model_exists, model_path = _check_model_exists(model_name)
        if not model_exists:
            return jsonify({
                'status': 'error',
                'message': f'Le modèle "{model_name}" n\'existe pas localement ou sur Hugging Face Hub.'
            }), 404
        
        # Trouver un GPU libre
        gpu_id = _find_free_gpu(manager)
        if gpu_id == -1:
            status = manager.get_model_status()
            loaded_models = []
            for gid in [0, 1]:
                gpu_info = status["gpus"].get(gid, {})
                if gpu_info.get("model_loaded", False):
                    loaded_models.append(f"GPU {gid}: {gpu_info.get('model_name', 'unknown')}")
            
            return jsonify({
                'status': 'error',
                'message': 'Aucun GPU libre. Tous les GPUs ont déjà un modèle chargé.',
                'loaded_models': loaded_models
            }), 503
        
        # Récupérer les paramètres optionnels depuis le body de la requête
        request_data = request.get_json() or {}
        model_kwargs = request_data.get('model_kwargs', {})
        
        # Charger le modèle
        logger.info(f"Chargement du modèle '{model_name}' sur GPU {gpu_id}...")
        success, access_token = manager.load_model(model_name, gpu_id=gpu_id, **model_kwargs)
        
        if success:
            # Récupérer le statut du GPU après chargement
            gpu_status = manager.get_model_status(gpu_id=gpu_id)
            
            return jsonify({
                'status': 'success',
                'message': f'Modèle "{model_name}" chargé avec succès sur GPU {gpu_id}',
                'gpu_id': gpu_id,
                'gpu_identifier': f'GPU-{gpu_id}',
                'model_name': model_name,
                'access_token': access_token,
                'note': 'Conservez ce token pour décharger le modèle plus tard',
                'gpu_status': gpu_status
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': f'Erreur lors du chargement du modèle "{model_name}" sur GPU {gpu_id}'
            }), 500
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur serveur: {str(e)}'
        }), 500


@model_management_bp.route('/models/unload/<int:gpu_id>', methods=['POST'])
def unload_model(gpu_id: int):
    """
    Décharge un modèle d'un GPU spécifique.
    
    Args:
        gpu_id: Numéro du GPU (0 ou 1) à décharger
    
    Body JSON requis:
        {
            "access_token": "token_reçu_lors_du_chargement"
        }
    
    Returns:
        - 200: Modèle déchargé avec succès
        - 400: Token manquant ou invalide
        - 404: Aucun modèle sur ce GPU
        - 500: Erreur lors du déchargement
    """
    try:
        manager = get_llm_manager()
        
        # Vérifier que le GPU ID est valide
        if gpu_id not in [0, 1]:
            return jsonify({
                'status': 'error',
                'message': f'GPU ID invalide: {gpu_id}. Doit être 0 ou 1.'
            }), 400
        
        # Récupérer le token depuis le body de la requête
        request_data = request.get_json() or {}
        access_token = request_data.get('access_token')
        
        if not access_token:
            return jsonify({
                'status': 'error',
                'message': 'Token d\'accès requis. Fournissez le token reçu lors du chargement du modèle dans le body JSON (access_token).'
            }), 400
        
        # Décharger le modèle
        logger.info(f"Tentative de déchargement du GPU {gpu_id}...")
        success, message = manager.unload_model(gpu_id, access_token=access_token)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'gpu_id': gpu_id,
                'gpu_identifier': f'GPU-{gpu_id}'
            }), 200
        else:
            # Déterminer le code de statut HTTP approprié
            status_code = 400  # Par défaut: mauvaise requête
            
            if "Token d'accès invalide" in message or "Token d'accès requis" in message:
                status_code = 403  # Forbidden
            elif "Aucun modèle chargé" in message:
                status_code = 404  # Not Found
            
            return jsonify({
                'status': 'error',
                'message': message,
                'gpu_id': gpu_id
            }), status_code
    
    except Exception as e:
        logger.error(f"Erreur lors du déchargement du modèle: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur serveur: {str(e)}'
        }), 500

