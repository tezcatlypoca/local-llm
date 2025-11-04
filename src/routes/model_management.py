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
from llm_gguf_manager_instance import get_gguf_manager

if TYPE_CHECKING:
    from llm_manager import LLMManager
    from llm_gguf_manager import LLMGGUFManager

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


def _find_free_gpu(manager, is_gguf: bool = False) -> int:
    """Trouve un GPU libre pour charger un modèle. Préfère GPU 1 (évite le GPU d'affichage)."""
    status = manager.get_model_status()

    # Chercher d'abord GPU 1, puis GPU 0 (priorité au GPU 1 pour éviter le GPU d'affichage)
    for gpu_id in [1, 0]:
        if is_gguf:
            # Pour GGUF, vérifier directement dans le statut
            if not status.get("gpus", {}).get(gpu_id, {}).get("model_loaded", False):
                return gpu_id
        else:
            # Pour transformers, vérifier gpu_available ET model_loaded
            gpu_info = status.get("gpus", {}).get(gpu_id, {})
            if gpu_info.get("gpu_available", False) and not gpu_info.get("model_loaded", False):
                return gpu_id

    return -1


def _is_gguf_model(model_name: str) -> bool:
    """Détecte si un modèle est au format GGUF."""
    model_name_lower = model_name.lower()
    # Vérifier si le nom contient .gguf ou si c'est un chemin vers un fichier .gguf
    if '.gguf' in model_name_lower or 'gguf' in model_name_lower:
        return True
    # Vérifier si c'est un chemin de fichier qui se termine par .gguf
    from pathlib import Path
    path = Path(model_name)
    if path.exists() and path.suffix.lower() == '.gguf':
        return True
    # Vérifier si c'est un identifiant qui correspond à un modèle GGUF dans le cache
    # (chercher dans les modèles listés par GET /models)
    try:
        from routes.models import _scan_huggingface_models
        models = _scan_huggingface_models()
        for model in models:
            if model.get("identifier") == model_name:
                if model.get("model_format") == "gguf":
                    return True
                # Si le modèle a des fichiers GGUF, c'est un modèle GGUF
                if model.get("gguf_files"):
                    return True
    except Exception:
        pass
    return False


@model_management_bp.route('/models/load/<path:model_name>', methods=['POST'])
def load_model(model_name: str):
    """Charge un modèle sur un GPU libre (transformers ou GGUF)."""
    try:
        # Détecter le type de modèle
        is_gguf = _is_gguf_model(model_name)
        
        # Utiliser le bon gestionnaire
        if is_gguf:
            manager = get_gguf_manager()
            logger.info(f"Modèle GGUF détecté: {model_name}")
        else:
            manager = get_llm_manager()
            logger.info(f"Modèle transformers détecté: {model_name}")

        # Vérifier que le modèle existe (pour GGUF, vérifier le fichier)
        if is_gguf:
            # Pour GGUF, model_name peut être un chemin de fichier ou un nom de modèle
            from pathlib import Path
            model_path_obj = Path(model_name)
            if model_path_obj.exists() and model_path_obj.suffix == '.gguf':
                model_exists = True
                model_path = model_name
            else:
                # Chercher dans le cache Hugging Face
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                model_path = None
                model_exists = False
                
                if cache_dir.exists():
                    # Méthode 1: Chercher directement le fichier .gguf
                    for gguf_file in cache_dir.rglob("*.gguf"):
                        if model_name.replace("/", "--") in str(gguf_file) or model_name in str(gguf_file) or any(part in str(gguf_file) for part in model_name.split("/")):
                            model_path = str(gguf_file)
                            model_exists = True
                            logger.info(f"Fichier GGUF trouvé: {model_path}")
                            break
                    
                    # Méthode 2: Si pas trouvé, chercher via GET /models
                    if not model_exists:
                        try:
                            from routes.models import _scan_huggingface_models
                            models = _scan_huggingface_models()
                            for model in models:
                                if model.get("identifier") == model_name or model.get("identifier", "").replace("/", "--") == model_name.replace("/", "--"):
                                    if model.get("model_format") == "gguf" and model.get("recommended_gguf_file"):
                                        # Construire le chemin complet
                                        model_dir = Path(model.get("path", ""))
                                        if model_dir.exists():
                                            gguf_file_path = model_dir / model.get("recommended_gguf_file")
                                            if gguf_file_path.exists():
                                                model_path = str(gguf_file_path)
                                                model_exists = True
                                                logger.info(f"Fichier GGUF trouvé via scan: {model_path}")
                                                break
                        except Exception as e:
                            logger.debug(f"Erreur lors de la recherche via scan: {e}")
        else:
            model_exists, model_path = _check_model_exists(model_name)
        
        if not model_exists:
            return jsonify({
                'status': 'error',
                'message': f'Le modèle "{model_name}" n\'existe pas localement ou n\'a pas été trouvé.'
            }), 404

        gpu_id = _find_free_gpu(manager, is_gguf=is_gguf)
        if gpu_id == -1:
            status = manager.get_model_status()
            loaded_models = []
            for gid in [0, 1]:
                if is_gguf:
                    gpu_info = status.get("gpus", {}).get(gid, {})
                else:
                    gpu_info = status["gpus"].get(gid, {})
                if gpu_info.get("model_loaded", False):
                    loaded_models.append(f"GPU {gid}: {gpu_info.get('model_name', 'unknown')}")

            return jsonify({
                'status': 'error',
                'message': 'Aucun GPU libre. Tous les GPUs ont déjà un modèle chargé.',
                'loaded_models': loaded_models
            }), 503

        # Récupérer les données JSON
        try:
            request_data = request.get_json(force=True, silent=True) or {}
        except Exception:
            request_data = {}
        model_kwargs = request_data.get('model_kwargs', {})
        
        # Pour GGUF, tokenizer_name peut être fourni
        tokenizer_name = request_data.get('tokenizer_name', None)

        logger.info(f"Chargement du modèle '{model_name}' (type: {'GGUF' if is_gguf else 'transformers'}) sur GPU {gpu_id}...")
        logger.info(f"Détection GGUF: {is_gguf}, Chemin modèle: {model_path}")
        
        try:
            if is_gguf:
                logger.info(f"Utilisation du gestionnaire GGUF pour charger: {model_path}")
                success, access_token = manager.load_model(
                    model_path, 
                    gpu_id=gpu_id, 
                    tokenizer_name=tokenizer_name,
                    **model_kwargs
                )
            else:
                logger.info(f"Utilisation du gestionnaire transformers pour charger: {model_name}")
                success, access_token = manager.load_model(model_name, gpu_id=gpu_id, **model_kwargs)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Erreur lors du chargement: {str(e)}',
                'model_type': 'gguf' if is_gguf else 'transformers'
            }), 500

        if success:
            gpu_status = manager.get_model_status(gpu_id=gpu_id)

            return jsonify({
                'status': 'success',
                'message': f'Modèle "{model_name}" chargé avec succès sur GPU {gpu_id}',
                'gpu_id': gpu_id,
                'gpu_identifier': f'GPU-{gpu_id}',
                'model_name': model_name,
                'model_path': model_path,
                'model_type': 'gguf' if is_gguf else 'transformers',
                'access_token': access_token,
                'note': 'Conservez ce token pour décharger le modèle plus tard',
                'gpu_status': gpu_status
            }), 200
        else:
            # Améliorer le message d'erreur avec plus de détails
            tips = []
            if is_gguf:
                tips = [
                    'Vérifiez que le fichier .gguf existe et est accessible',
                    'Assurez-vous que llama-cpp-python est installé: pip install llama-cpp-python',
                    'Pour ROCm/AMD, installez avec: CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python',
                    'Vérifiez que vous avez suffisamment de mémoire GPU disponible'
                ]
            else:
                tips = [
                    'Utilisez l\'identifier exact du modèle depuis GET /models/',
                    'Assurez-vous que le modèle est compatible avec transformers (format .bin, .safetensors)',
                    'Vérifiez que vous avez suffisamment de mémoire GPU disponible',
                    'Pour les modèles Llama, assurez-vous d\'avoir les bons tokens spéciaux configurés'
                ]
            
            return jsonify({
                'status': 'error',
                'message': f'Erreur lors du chargement du modèle "{model_name}" sur GPU {gpu_id}',
                'model_type': 'gguf' if is_gguf else 'transformers',
                'details': 'Vérifiez les logs du serveur pour plus d\'informations.',
                'tips': tips
            }), 500

    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur serveur: {str(e)}'
        }), 500


@model_management_bp.route('/models/unload/<int:gpu_id>', methods=['POST'])
def unload_model(gpu_id: int):
    """Décharge un modèle d'un GPU spécifique (transformers ou GGUF)."""
    try:
        if gpu_id not in [0, 1]:
            return jsonify({
                'status': 'error',
                'message': f'GPU ID invalide: {gpu_id}. Doit être 0 ou 1.'
            }), 400

        # Récupérer les données JSON
        try:
            request_data = request.get_json(force=True, silent=True) or {}
        except Exception:
            request_data = {}
        access_token = request_data.get('access_token')

        if not access_token:
            return jsonify({
                'status': 'error',
                'message': "Token d'accès requis. Fournissez le token reçu lors du chargement du modèle (access_token)."
            }), 400

        # Détecter le type de modèle en vérifiant les deux gestionnaires
        transformers_manager = get_llm_manager()
        gguf_manager = get_gguf_manager()
        
        transformers_status = transformers_manager.get_model_status(gpu_id=gpu_id)
        gguf_status = gguf_manager.get_model_status(gpu_id=gpu_id)
        
        # Utiliser le gestionnaire qui a un modèle chargé
        if transformers_status.get("model_loaded", False):
            manager = transformers_manager
            model_type = "transformers"
        elif gguf_status.get("model_loaded", False):
            manager = gguf_manager
            model_type = "gguf"
        else:
            return jsonify({
                'status': 'error',
                'message': f'Aucun modèle chargé sur GPU {gpu_id}.'
            }), 404

        logger.info(f"Tentative de déchargement du GPU {gpu_id} (type: {model_type})...")
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


@model_management_bp.route('/models/unload_all', methods=['POST'])
def unload_all_models():
    """Décharge tous les modèles de tous les GPUs (transformers et GGUF). Requiert un mot de passe administrateur."""
    try:
        # Récupérer les données JSON
        try:
            request_data = request.get_json(force=True, silent=True) or {}
        except Exception:
            request_data = {}
        
        admin_password = request_data.get('password')
        
        if not admin_password:
            return jsonify({
                'status': 'error',
                'message': "Mot de passe administrateur requis. Fournissez le champ 'password' dans le body de la requête."
            }), 400

        logger.info(f"Tentative de déchargement forcé de tous les GPUs...")
        
        # Décharger depuis les deux gestionnaires
        transformers_manager = get_llm_manager()
        gguf_manager = get_gguf_manager()
        
        transformers_results = transformers_manager.unload_all_models(admin_password=admin_password)
        
        # Pour GGUF, on doit décharger manuellement car il n'y a pas de unload_all_models
        gguf_results = {"gpu_0": {"success": False, "message": ""}, "gpu_1": {"success": False, "message": ""}}
        for gpu_id in [0, 1]:
            gguf_status = gguf_manager.get_model_status(gpu_id=gpu_id)
            if gguf_status.get("model_loaded", False):
                # Pas de force unload pour GGUF, on essaie juste de nettoyer
                try:
                    gguf_manager.models[gpu_id] = None
                    gguf_manager.tokenizers[gpu_id] = None
                    gguf_manager.model_paths[gpu_id] = None
                    gguf_manager.model_names[gpu_id] = None
                    gguf_manager.tokenizer_names[gpu_id] = None
                    gguf_manager.access_tokens[gpu_id] = None
                    import gc
                    gc.collect()
                    gguf_results[f"gpu_{gpu_id}"] = {"success": True, "message": f"Modèle GGUF déchargé du GPU {gpu_id}"}
                except Exception as e:
                    gguf_results[f"gpu_{gpu_id}"] = {"success": False, "message": f"Erreur: {str(e)}"}
            else:
                gguf_results[f"gpu_{gpu_id}"] = {"success": True, "message": f"Aucun modèle GGUF chargé sur GPU {gpu_id}"}
        
        # Combiner les résultats
        results = {
            "gpu_0": {
                "success": transformers_results["gpu_0"]["success"] or gguf_results["gpu_0"]["success"],
                "message": f"Transformers: {transformers_results['gpu_0']['message']}. GGUF: {gguf_results['gpu_0']['message']}"
            },
            "gpu_1": {
                "success": transformers_results["gpu_1"]["success"] or gguf_results["gpu_1"]["success"],
                "message": f"Transformers: {transformers_results['gpu_1']['message']}. GGUF: {gguf_results['gpu_1']['message']}"
            }
        }

        # Vérifier si le mot de passe était invalide (vérifier dans les deux GPUs au cas où)
        invalid_password_msg = "Mot de passe administrateur invalide"
        for gpu_key in ["gpu_0", "gpu_1"]:
            if not results[gpu_key]["success"] and invalid_password_msg in results[gpu_key]["message"]:
                return jsonify({
                    'status': 'error',
                    'message': results[gpu_key]["message"]
                }), 403

        # Compter les succès
        success_count = sum(1 for gpu_result in results.values() if gpu_result["success"])
        total_gpus = len(results)

        # Préparer la réponse détaillée
        response_data = {
            'status': 'success' if success_count == total_gpus else 'partial',
            'message': f'Déchargement terminé: {success_count}/{total_gpus} GPUs traités avec succès',
            'results': {
                'gpu_0': {
                    'gpu_id': 0,
                    'gpu_identifier': 'GPU-0',
                    'success': results["gpu_0"]["success"],
                    'message': results["gpu_0"]["message"]
                },
                'gpu_1': {
                    'gpu_id': 1,
                    'gpu_identifier': 'GPU-1',
                    'success': results["gpu_1"]["success"],
                    'message': results["gpu_1"]["message"]
                }
            }
        }

        status_code = 200
        if success_count == 0:
            response_data['status'] = 'error'
            status_code = 500

        return jsonify(response_data), status_code

    except Exception as e:
        logger.error(f"Erreur lors du déchargement de tous les modèles: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur serveur: {str(e)}'
        }), 500