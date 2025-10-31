"""
Routes pour la gestion des modèles disponibles.
"""
import os
import json
from pathlib import Path
from flask import Blueprint, jsonify
import logging

logger = logging.getLogger(__name__)

# Création du Blueprint
models_bp = Blueprint('models', __name__)

try:
    from huggingface_hub import scan_cache_dir
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub non disponible, scan manuel du cache")


def _get_huggingface_cache_dir():
    """Retourne le répertoire du cache Hugging Face."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    return cache_dir


def _calculate_directory_size(directory):
    """Calcule la taille totale d'un répertoire en octets."""
    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logger.debug(f"Erreur calcul taille: {e}")
    return total_size


def _scan_huggingface_models():
    """Scanne le cache Hugging Face pour trouver les modèles téléchargés."""
    models = []

    if HF_HUB_AVAILABLE:
        try:
            cache_info = scan_cache_dir()
            for repo in list(cache_info.repos):
                revisions_list = list(repo.revisions) if repo.revisions else []
                if not revisions_list:
                    continue

                latest_revision = revisions_list[-1]

                # Taille disque
                size_bytes = 0
                if hasattr(latest_revision, "size_on_disk"):
                    size_val = latest_revision.size_on_disk
                    if isinstance(size_val, (int, float)):
                        size_bytes = size_val
                    elif isinstance(size_val, str):
                        try:
                            size_bytes = int(size_val)
                        except (ValueError, TypeError):
                            size_bytes = 0

                # Noms de fichiers
                file_names = []
                if hasattr(latest_revision, "files") and latest_revision.files:
                    for f in list(latest_revision.files):
                        for attr in ("file_path", "blob_path", "filename", "file_name", "path"):
                            if hasattr(f, attr):
                                path_val = getattr(f, attr)
                                if path_val:
                                    file_names.append(os.path.basename(str(path_val)))
                                    break

                model_info = {
                    "identifier": repo.repo_id,
                    "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes > 0 else 0,
                    "revision": getattr(latest_revision, "commit_hash", None),
                    "path": str(getattr(latest_revision, "snapshot_path", "")),
                    "files": file_names,
                }

                # Lecture du fichier config.json si présent
                if model_info["path"]:
                    config_path = os.path.join(model_info["path"], "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, "r", encoding="utf-8") as f:
                                config = json.load(f)
                                model_info["model_type"] = config.get("model_type", "unknown")
                                model_info["architectures"] = config.get("architectures", [])
                        except Exception as e:
                            logger.debug(f"Erreur lecture config.json: {e}")

                models.append(model_info)

            logger.info(f"Scan Hugging Face Hub: {len(models)} modèle(s) trouvé(s)")
            return models

        except Exception as e:
            logger.warning(f"Erreur avec scan_cache_dir(), fallback sur scan manuel: {e}")

    # --- Méthode de secours : scan manuel ---
    cache_dir = _get_huggingface_cache_dir()

    if not os.path.exists(cache_dir):
        logger.debug(f"Cache Hugging Face introuvable: {cache_dir}")
        return models

    try:
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if not os.path.isdir(item_path) or item.startswith("."):
                continue

            has_model_files = False
            config_path = None
            model_info = {
                "path": str(item_path),
                "files": []
            }

            for root, _, files in os.walk(item_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, item_path)

                    if file.endswith((".bin", ".safetensors", ".pt", ".pth", ".gguf")):
                        has_model_files = True
                        model_info["files"].append(rel_path)
                    elif file == "config.json":
                        config_path = os.path.join(root, file)
                    elif file in ("tokenizer.json", "vocab.json", "merges.txt"):
                        model_info.setdefault("tokenizer_files", []).append(rel_path)

            if has_model_files:
                if config_path and os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                            model_info["model_type"] = config.get("model_type", "unknown")
                            model_info["architectures"] = config.get("architectures", [])
                    except Exception as e:
                        logger.debug(f"Erreur lecture config.json: {e}")

                if "--" in item and item.startswith("models--"):
                    parts = item.split("--")
                    if len(parts) >= 3:
                        org = parts[1]
                        model_name = parts[2]
                        model_info["identifier"] = f"{org}/{model_name}"
                        model_info["organization"] = org
                        model_info["name"] = model_name
                    else:
                        model_info["identifier"] = item
                else:
                    model_info["identifier"] = item

                model_info["size_mb"] = round(_calculate_directory_size(item_path) / (1024 * 1024), 2)
                models.append(model_info)

        logger.info(f"Scan manuel du cache: {len(models)} modèle(s) trouvé(s)")

    except Exception as e:
        logger.error(f"Erreur lors du scan du cache Hugging Face: {e}", exc_info=True)

    return models


def _scan_local_models_directory(local_dir: str = None):
    """Scanne un répertoire local pour trouver des modèles."""
    models = []

    if local_dir is None:
        project_root = Path(__file__).parent.parent.parent
        local_dir = project_root / "models"

    local_dir = Path(local_dir)

    if not local_dir.exists() or not local_dir.is_dir():
        logger.debug(f"Dossier de modèles locaux introuvable: {local_dir}")
        return models

    try:
        for item in local_dir.iterdir():
            if item.is_dir():
                model_info = {
                    "identifier": item.name,
                    "path": str(item),
                    "local": True,
                    "files": []
                }

                for root, _, files in os.walk(item):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), item)
                        if file.endswith((".bin", ".safetensors", ".pt", ".pth", ".gguf")):
                            model_info["files"].append(rel_path)
                        elif file == "config.json":
                            config_path = os.path.join(root, file)
                            try:
                                with open(config_path, "r", encoding="utf-8") as f:
                                    config = json.load(f)
                                    model_info["model_type"] = config.get("model_type", "unknown")
                                    model_info["architectures"] = config.get("architectures", [])
                            except Exception as e:
                                logger.debug(f"Erreur lecture config.json: {e}")

                if model_info["files"]:
                    model_info["size_mb"] = round(_calculate_directory_size(item) / (1024 * 1024), 2)
                    models.append(model_info)

    except Exception as e:
        logger.error(f"Erreur lors du scan du dossier local: {e}", exc_info=True)

    return models


@models_bp.route('/models', methods=['GET'])
def list_models():
    """
    Liste tous les modèles disponibles en local et dans le cache Hugging Face.
    """
    try:
        models = []
        hf_models = _scan_huggingface_models()
        models.extend(hf_models)

        local_models = _scan_local_models_directory()
        models.extend(local_models)

        cache_dir = _get_huggingface_cache_dir()

        # Conversion sûre en JSON, même si un objet inattendu apparaît
        response = {
            "status": "success",
            "count": len(models),
            "cache_dir": str(cache_dir),
            "cache_exists": os.path.exists(cache_dir),
            "models": models,
        }

        return jsonify(json.loads(json.dumps(response, default=str))), 200

    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la liste des modèles: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
