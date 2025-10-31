"""
Routes pour la gestion du chargement/déchargement des modèles sur GPU.
"""
import os
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request
import sys
from typing import TYPE_CHECKING

# Import du dossier parent
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from llm_manager_instance import get_llm_manager

if TYPE_CHECKING:
    from llm_manager import LLMManager

logger = logging.getLogger(__name__)

# Création du Blueprint
model_management_bp = Blueprint('model_management', __name__)


def _check_model_exists(model_name: str) -> tuple[bool, str]:
    """
    Vérifie si un modèle existe localement sur la machine ou dans le cache Hugging Face.
    """
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in list(cache_info.repos):
            if repo.repo_id == model_name:
                if repo.revisions:
                    # ✅ Correction : convertir en liste pour indexation
                    revisions = list(repo.revisions)
                    if not revisions:
                        return True, None

                    latest_revision = revisions[-1]

                    # Certaines versions n'ont pas snapshot_path
                    snapshot_path = getattr(latest_revision, "snapshot_path", None)
                    if snapshot_path:
                        return True, str(snapshot_path)
                    else:
                        return True, None

        # Vérifier aussi dans un dossier local si configuré
        project_root = Path(__file__).parent.parent.parent
        local_models_dir = project_root / "models"

        if local_models_dir.exists():
            model_dir = local_models_dir / model_name.replace("/", "_")
            if model_dir.exists() and model_dir.is_dir():
                # Vérifier qu'il y a des fichiers de modèle
                for root, _, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.gguf')):
                            return True, str(model_dir)

        return False, None

    except Exception as e:
        logger.error(f"Erreur lors de la vérification du modèle: {e}", exc_info=True)
        return False, None


def _find_free_gpu(manager) -> int:
    """Trouve un GPU libre pour charger un modèle."""
    status = manager.get_model_status()

    for gpu_id in [0, 1]:
        gpu_info = status["gpus"].get(gpu_id, {})
        if gpu_info.get("gpu_available", False) and not gpu_info.get("model_loaded", False):
            return gpu_id

    return -1


@model_management_bp.route('/models/load/<path:model_name>', methods=['POST'])
def load_model(model_name: str):
    """Charge un modèle sur un GPU libre."""
    try:
        manager = get_llm_manager()

        # Vérifier que le modèle existe
        model_exists, model_path = _check_model_exists(model_name)
        if not model_exists:
            return jsonify({
                'status': 'error',
                'message': f'Le modèle "{model_name}" n\'existe pas localement ou sur Hugging Face Hub.'
            }), 404

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

        request_data = request.get_json() or {}
        model_kwargs = request_data.get('model_kwargs', {})

        logger.info(f"Chargement du modèle '{model_name}' sur GPU {gpu_id}...")
        success, access_token = manager.load_model(model_name, gpu_id=gpu_id, **model_kwargs)

        if success:
            gpu_status = manager.get_model_status(gpu_id=gpu_id)

            return jsonify({
                'status': 'success',
                'message': f'Modèle "{model_name}" chargé avec succès sur GPU {gpu_id}',
                'gpu_id': gpu_id,
                'gpu_identifier': f'GPU-{gpu_id}',
                'model_name': model_name,
                'model_path': model_path,
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
    """Décharge un modèle d'un GPU spécifique."""
    try:
        manager = get_llm_manager()

        if gpu_id not in [0, 1]:
            return jsonify({
                'status': 'error',
                'message': f'GPU ID invalide: {gpu_id}. Doit être 0 ou 1.'
            }), 400

        request_data = request.get_json() or {}
        access_token = request_data.get('access_token')

        if not access_token:
            return jsonify({
                'status': 'error',
                'message': "Token d'accès requis. Fournissez le token reçu lors du chargement du modèle (access_token)."
            }), 400

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
            status_code = 400
            if "Token d'accès invalide" in message or "Token d'accès requis" in message:
                status_code = 403
            elif "Aucun modèle chargé" in message:
                status_code = 404

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
