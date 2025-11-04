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
                gguf_files = []
                if hasattr(latest_revision, "files") and latest_revision.files:
                    for f in list(latest_revision.files):
                        for attr in ("file_path", "blob_path", "filename", "file_name", "path"):
                            if hasattr(f, attr):
                                path_val = getattr(f, attr)
                                if path_val:
                                    filename = os.path.basename(str(path_val))
                                    file_names.append(filename)
                                    # Détecter les fichiers GGUF
                                    if filename.endswith('.gguf'):
                                        gguf_files.append(filename)
                                    break

                model_info = {
                    "identifier": repo.repo_id,
                    "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes > 0 else 0,
                    "revision": getattr(latest_revision, "commit_hash", None),
                    "path": str(getattr(latest_revision, "snapshot_path", "")),
                    "files": file_names,
                }
                
                # Détecter le format du modèle
                # Vérifier d'abord dans les fichiers si pas trouvé dans les métadonnées
                snapshot_path = getattr(latest_revision, "snapshot_path", None)
                if snapshot_path and os.path.exists(snapshot_path):
                    for root, _, files in os.walk(snapshot_path):
                        for file in files:
                            if file.endswith('.gguf'):
                                if file not in gguf_files:
                                    gguf_files.append(file)
                
                if gguf_files:
                    model_info["model_format"] = "gguf"
                    model_info["gguf_files"] = sorted(gguf_files)
                    # Recommander le meilleur fichier GGUF (préférer Q4_K_M, puis Q4_0)
                    recommended = None
                    for gguf in sorted(gguf_files):
                        if 'q4_k_m' in gguf.lower():
                            recommended = gguf
                            break
                    if not recommended:
                        for gguf in sorted(gguf_files):
                            if 'q4_0' in gguf.lower() or 'q4' in gguf.lower():
                                recommended = gguf
                                break
                    if not recommended and gguf_files:
                        recommended = gguf_files[0]
                    model_info["recommended_gguf_file"] = recommended
                else:
                    model_info["model_format"] = "transformers"

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
        # D'abord, scanner tous les fichiers .gguf dans le cache pour trouver les modèles GGUF
        # (parfois ils sont dans des sous-répertoires qui ne sont pas des snapshots)
        all_gguf_files = {}
        for root, _, files in os.walk(cache_dir):
            for file in files:
                if file.endswith('.gguf'):
                    full_path = os.path.join(root, file)
                    # Extraire l'identifiant depuis le chemin
                    rel_path = os.path.relpath(full_path, cache_dir)
                    parts = rel_path.split(os.sep)
                    
                    model_id = None
                    # Méthode 1: Chercher models--org--model
                    for part in parts:
                        if part.startswith("models--") and "--" in part:
                            model_id = part.replace("models--", "").replace("--", "/")
                            break
                    
                    # Méthode 2: Si pas trouvé, chercher des patterns avec espaces (ex: "bartowski - Qwen2.5 - 7B - Instruct - GGUF")
                    if not model_id:
                        for part in parts:
                            # Détecter les patterns comme "bartowski - Qwen2.5 - 7B - Instruct - GGUF"
                            if " - " in part or "- " in part:
                                # Essayer de reconstruire l'identifiant
                                # Ex: "bartowski - Qwen2.5 - 7B - Instruct - GGUF" -> "bartowski/Qwen2.5-7B-Instruct-GGUF"
                                clean_part = part.replace(" - ", "/").replace("- ", "-").replace(" -", "-")
                                # Nettoyer les espaces restants
                                clean_part = clean_part.replace(" ", "-")
                                if "/" in clean_part:
                                    model_id = clean_part
                                    break
                            # Ou chercher directement dans les noms de répertoires
                            elif any(keyword in part.lower() for keyword in ["qwen", "mistral", "codellama", "llama"]):
                                # Essayer de trouver l'org dans les parties précédentes
                                for i, p in enumerate(parts):
                                    if p == part and i > 0:
                                        # Prendre la partie précédente comme org
                                        org = parts[i-1].replace(" ", "").replace("-", "")
                                        model_name = part.replace(" ", "-").replace("_", "-")
                                        model_id = f"{org}/{model_name}"
                                        break
                                if model_id:
                                    break
                    
                    if model_id:
                        if model_id not in all_gguf_files:
                            all_gguf_files[model_id] = []
                        all_gguf_files[model_id].append({
                            "file": file,
                            "path": full_path,
                            "dir": root
                        })
        
        # Maintenant scanner les répertoires normalement
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

            gguf_files = []
            gguf_paths = {}  # Stocker les chemins complets des fichiers GGUF
            for root, _, files in os.walk(item_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, item_path)

                    if file.endswith((".bin", ".safetensors", ".pt", ".pth", ".gguf")):
                        has_model_files = True
                        model_info["files"].append(rel_path)
                        if file.endswith('.gguf'):
                            gguf_files.append(file)
                            gguf_paths[file] = full_path
                    elif file == "config.json":
                        config_path = os.path.join(root, file)
                    elif file in ("tokenizer.json", "vocab.json", "merges.txt"):
                        model_info.setdefault("tokenizer_files", []).append(rel_path)
            
            # Détecter le format du modèle
            if gguf_files:
                model_info["model_format"] = "gguf"
                model_info["gguf_files"] = sorted(gguf_files)
                # Recommander le meilleur fichier GGUF
                recommended = None
                for gguf in sorted(gguf_files):
                    if 'q4_k_m' in gguf.lower():
                        recommended = gguf
                        break
                if not recommended:
                    for gguf in sorted(gguf_files):
                        if 'q4_0' in gguf.lower() or 'q4' in gguf.lower():
                            recommended = gguf
                            break
                if not recommended and gguf_files:
                    recommended = gguf_files[0]
                model_info["recommended_gguf_file"] = recommended
                # Stocker le chemin complet du fichier recommandé
                if recommended and recommended in gguf_paths:
                    model_info["recommended_gguf_file_path"] = gguf_paths[recommended]
            else:
                model_info["model_format"] = "transformers"

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
        
        # Ajouter les modèles GGUF trouvés qui ne sont pas dans les répertoires principaux
        for model_id, gguf_list in all_gguf_files.items():
            # Vérifier si ce modèle n'est pas déjà dans la liste
            existing = False
            for existing_model in models:
                if existing_model.get("identifier") == model_id:
                    existing = True
                    break
            
            if not existing and gguf_list:
                # Créer une entrée pour ce modèle GGUF
                gguf_file_names = [g["file"] for g in gguf_list]
                recommended = None
                for gguf in sorted(gguf_file_names):
                    if 'q4_k_m' in gguf.lower():
                        recommended = gguf
                        break
                if not recommended:
                    for gguf in sorted(gguf_file_names):
                        if 'q4_0' in gguf.lower() or 'q4' in gguf.lower():
                            recommended = gguf
                            break
                if not recommended:
                    recommended = gguf_file_names[0]
                
                # Trouver le chemin du fichier recommandé
                recommended_path = None
                for g in gguf_list:
                    if g["file"] == recommended:
                        recommended_path = g["path"]
                        break
                
                model_info = {
                    "identifier": model_id,
                    "model_format": "gguf",
                    "gguf_files": sorted(gguf_file_names),
                    "recommended_gguf_file": recommended,
                    "recommended_gguf_file_path": recommended_path,
                    "path": os.path.dirname(gguf_list[0]["path"]) if gguf_list else None,
                    "files": gguf_file_names,
                    "size_mb": round(sum(os.path.getsize(g["path"]) for g in gguf_list) / (1024 * 1024), 2) if gguf_list else 0
                }
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

                gguf_files = []
                for root, _, files in os.walk(item):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), item)
                        if file.endswith((".bin", ".safetensors", ".pt", ".pth", ".gguf")):
                            model_info["files"].append(rel_path)
                            if file.endswith('.gguf'):
                                gguf_files.append(file)
                        elif file == "config.json":
                            config_path = os.path.join(root, file)
                            try:
                                with open(config_path, "r", encoding="utf-8") as f:
                                    config = json.load(f)
                                    model_info["model_type"] = config.get("model_type", "unknown")
                                    model_info["architectures"] = config.get("architectures", [])
                            except Exception as e:
                                logger.debug(f"Erreur lecture config.json: {e}")
                
                # Détecter le format du modèle
                if gguf_files:
                    model_info["model_format"] = "gguf"
                    model_info["gguf_files"] = sorted(gguf_files)
                    # Recommander le meilleur fichier GGUF
                    recommended = None
                    for gguf in sorted(gguf_files):
                        if 'q4_k_m' in gguf.lower():
                            recommended = gguf
                            break
                    if not recommended:
                        for gguf in sorted(gguf_files):
                            if 'q4_0' in gguf.lower() or 'q4' in gguf.lower():
                                recommended = gguf
                                break
                    if not recommended and gguf_files:
                        recommended = gguf_files[0]
                    model_info["recommended_gguf_file"] = recommended
                else:
                    model_info["model_format"] = "transformers"

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


@models_bp.route('/models/gguf', methods=['GET'])
def list_gguf_models():
    """
    Liste tous les modèles GGUF disponibles avec leurs chemins.
    """
    try:
        models = _scan_huggingface_models()
        local_models = _scan_local_models_directory()
        all_models = models + local_models
        
        gguf_models = []
        for model in all_models:
            if model.get("model_format") == "gguf" or model.get("gguf_files"):
                gguf_info = {
                    "identifier": model.get("identifier"),
                    "path": model.get("path"),
                    "gguf_files": model.get("gguf_files", []),
                    "recommended_gguf_file": model.get("recommended_gguf_file"),
                    "size_mb": model.get("size_mb", 0)
                }
                
                # Ajouter le chemin complet du fichier recommandé
                if gguf_info["recommended_gguf_file"] and gguf_info["path"]:
                    from pathlib import Path
                    model_dir = Path(gguf_info["path"])
                    if model_dir.exists():
                        full_path = model_dir / gguf_info["recommended_gguf_file"]
                        if full_path.exists():
                            gguf_info["recommended_gguf_file_path"] = str(full_path)
                
                gguf_models.append(gguf_info)
        
        return jsonify({
            "status": "success",
            "count": len(gguf_models),
            "gguf_models": gguf_models
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des modèles GGUF: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
