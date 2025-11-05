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


def _scan_all_gguf_files():
    """
    Scanne récursivement le cache Hugging Face pour trouver TOUS les fichiers .gguf.
    Retourne une liste de modèles avec identifiants uniques.
    """
    gguf_models = []
    cache_dir = Path(_get_huggingface_cache_dir())
    
    if not cache_dir.exists():
        logger.warning(f"Cache Hugging Face introuvable: {cache_dir}")
        return gguf_models
    
    # Dictionnaire pour regrouper les fichiers GGUF par modèle (basé sur le chemin)
    models_dict = {}
    
    try:
        # Scanner récursivement tous les fichiers .gguf
        for gguf_file in cache_dir.rglob("*.gguf"):
            try:
                full_path = str(gguf_file.absolute())
                file_name = gguf_file.name
                file_size = gguf_file.stat().st_size
                
                # Essayer d'extraire un identifiant depuis le chemin
                # Format typique: ~/.cache/huggingface/hub/models--org--model-name/snapshots/...
                parts = gguf_file.parts
                model_identifier = None
                
                # Chercher un répertoire qui commence par "models--"
                for i, part in enumerate(parts):
                    if part.startswith("models--"):
                        # Extraire org/model-name depuis "models--org--model-name"
                        model_part = part.replace("models--", "")
                        if "--" in model_part:
                            org, model_name = model_part.split("--", 1)
                            model_identifier = f"{org}/{model_name}"
                        break
                
                # Si pas d'identifiant trouvé, créer un identifiant basé sur le chemin
                if not model_identifier:
                    # Prendre le répertoire parent du fichier comme identifiant
                    parent_dir = gguf_file.parent.name
                    if parent_dir.startswith("models--"):
                        model_identifier = parent_dir.replace("models--", "").replace("--", "/")
                    else:
                        # Créer un identifiant basé sur le chemin relatif
                        rel_path = gguf_file.relative_to(cache_dir)
                        # Prendre les 2-3 premiers niveaux du chemin
                        path_parts = rel_path.parts[:3]
                        model_identifier = "/".join(path_parts).replace("--", "/")
                
                # Créer un identifiant unique pour ce fichier spécifique
                # Format: model_identifier/filename (sans extension)
                file_base = file_name.replace(".gguf", "")
                unique_id = f"{model_identifier}/{file_base}"
                
                # Regrouper les fichiers par modèle
                if model_identifier not in models_dict:
                    models_dict[model_identifier] = {
                        "identifier": model_identifier,
                        "gguf_files": [],
                        "paths": {}
                    }
                
                models_dict[model_identifier]["gguf_files"].append({
                    "file": file_name,
                    "unique_id": unique_id,
                    "path": full_path,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                })
                models_dict[model_identifier]["paths"][file_name] = full_path
                
            except Exception as e:
                logger.debug(f"Erreur lors du traitement de {gguf_file}: {e}")
                continue
        
        # Convertir en liste et déterminer le fichier recommandé
        for model_id, model_data in models_dict.items():
            gguf_files = model_data["gguf_files"]
            
            # Trier par taille et priorité de quantification
            def sort_key(f):
                priority = 999
                name_lower = f["file"].lower()
                if 'q4_k_m' in name_lower:
                    priority = 1
                elif 'q4_0' in name_lower or 'q4' in name_lower:
                    priority = 2
                elif 'q5_k_m' in name_lower:
                    priority = 3
                elif 'q5_0' in name_lower or 'q5' in name_lower:
                    priority = 4
                elif 'q8_0' in name_lower:
                    priority = 5
                return (priority, f["file"])
            
            gguf_files.sort(key=sort_key)
            
            # Fichier recommandé (le premier après tri)
            recommended = gguf_files[0] if gguf_files else None
            
            # Créer l'entrée du modèle
            model_info = {
                "identifier": model_id,
                "unique_identifiers": [f["unique_id"] for f in gguf_files],
                "gguf_files": [f["file"] for f in gguf_files],
                "recommended_gguf_file": recommended["file"] if recommended else None,
                "recommended_gguf_file_path": recommended["path"] if recommended else None,
                "recommended_gguf_unique_id": recommended["unique_id"] if recommended else None,
                "path": os.path.dirname(recommended["path"]) if recommended else None,
                "size_mb": sum(f["size_mb"] for f in gguf_files),
                "model_format": "gguf",
                "files_detail": gguf_files  # Détails complets pour chaque fichier
            }
            
            gguf_models.append(model_info)
        
        logger.info(f"Scan GGUF: {len(gguf_models)} modèle(s) trouvé(s) avec {sum(len(m['gguf_files']) for m in gguf_models)} fichier(s) .gguf")
        
    except Exception as e:
        logger.error(f"Erreur lors du scan des fichiers GGUF: {e}", exc_info=True)
    
    return gguf_models


def _resolve_gguf_identifier(identifier: str):
    """
    Résout un identifiant de modèle GGUF vers le chemin complet du fichier.
    
    Args:
        identifier: Identifiant du modèle (peut être l'identifier, unique_id, ou un chemin)
    
    Returns:
        (success: bool, file_path: str or None, error_message: str or None)
    """
    try:
        # Si c'est déjà un chemin qui existe, le retourner directement
        path_obj = Path(identifier)
        if path_obj.exists() and path_obj.suffix == '.gguf':
            return True, str(path_obj.absolute()), None
        
        # Scanner tous les modèles GGUF
        gguf_models = _scan_all_gguf_files()
        
        # Chercher par identifier, unique_id, ou nom de fichier
        for model in gguf_models:
            # Vérifier l'identifier principal
            if model.get("identifier") == identifier:
                recommended_path = model.get("recommended_gguf_file_path")
                if recommended_path and Path(recommended_path).exists():
                    return True, recommended_path, None
            
            # Vérifier les unique_identifiers
            unique_ids = model.get("unique_identifiers", [])
            if identifier in unique_ids:
                # Trouver le fichier correspondant
                files_detail = model.get("files_detail", [])
                for f in files_detail:
                    if f.get("unique_id") == identifier:
                        file_path = f.get("path")
                        if file_path and Path(file_path).exists():
                            return True, file_path, None
            
            # Vérifier par nom de fichier (sans extension)
            files_detail = model.get("files_detail", [])
            for f in files_detail:
                file_name = f.get("file", "")
                file_base = file_name.replace(".gguf", "")
                if identifier == file_base or identifier in file_name:
                    file_path = f.get("path")
                    if file_path and Path(file_path).exists():
                        return True, file_path, None
        
        # Chercher aussi dans les modèles scannés par l'ancienne méthode
        models = _scan_huggingface_models()
        for model in models:
            if model.get("model_format") == "gguf":
                if model.get("identifier") == identifier:
                    recommended_path = model.get("recommended_gguf_file_path")
                    if recommended_path and Path(recommended_path).exists():
                        return True, recommended_path, None
                
                # Chercher dans les fichiers GGUF du modèle
                gguf_files = model.get("gguf_files", [])
                for gguf_file in gguf_files:
                    if identifier in gguf_file or identifier.replace(".gguf", "") == gguf_file.replace(".gguf", ""):
                        model_dir = Path(model.get("path", ""))
                        if model_dir.exists():
                            full_path = model_dir / gguf_file
                            if full_path.exists():
                                return True, str(full_path), None
        
        # Dernière tentative: chercher récursivement dans le cache
        cache_dir = Path(_get_huggingface_cache_dir())
        if cache_dir.exists():
            # Chercher par nom de fichier
            for gguf_file in cache_dir.rglob(f"*{identifier}*.gguf"):
                if gguf_file.exists():
                    return True, str(gguf_file.absolute()), None
        
        return False, None, f"Identifiant '{identifier}' non trouvé dans les modèles GGUF disponibles"
    
    except Exception as e:
        logger.error(f"Erreur lors de la résolution de l'identifiant '{identifier}': {e}", exc_info=True)
        return False, None, f"Erreur lors de la résolution: {str(e)}"


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
                        for i, part in enumerate(parts):
                            # Détecter les patterns comme "bartowski - Qwen2.5 - 7B - Instruct - GGUF"
                            if " - " in part or "- " in part or " -" in part:
                                # Essayer de reconstruire l'identifiant
                                # Ex: "bartowski - Qwen2.5 - 7B - Instruct - GGUF" -> "bartowski/Qwen2.5-7B-Instruct-GGUF"
                                clean_part = part.replace(" - ", "/").replace("- ", "-").replace(" -", "-").replace(" -", "-")
                                # Nettoyer les espaces restants
                                clean_part = clean_part.replace(" ", "-").replace("_", "-")
                                if "/" in clean_part:
                                    model_id = clean_part
                                    break
                                # Si pas de "/", essayer de le construire
                                elif i > 0:
                                    # Prendre la partie précédente comme org potentiel
                                    org_part = parts[i-1].replace(" ", "").replace("-", "").replace("_", "")
                                    model_part = part.replace(" - ", "-").replace("- ", "-").replace(" -", "-").replace(" ", "-").replace("_", "-")
                                    if org_part and model_part:
                                        model_id = f"{org_part}/{model_part}"
                                        break
                            # Ou chercher directement dans les noms de répertoires contenant qwen/mistral/etc
                            elif any(keyword in part.lower() for keyword in ["qwen", "mistral", "codellama", "llama"]):
                                # Essayer de trouver l'org dans les parties précédentes
                                if i > 0:
                                    # Prendre la partie précédente comme org
                                    org = parts[i-1].replace(" ", "").replace("-", "").replace("_", "")
                                    model_name = part.replace(" ", "-").replace("_", "-").replace(" - ", "-").replace("- ", "-")
                                    if org and model_name:
                                        model_id = f"{org}/{model_name}"
                                        break
                                # Ou chercher dans le nom du fichier lui-même
                                if not model_id and "qwen" in file.lower():
                                    if "2.5" in file.lower() or "qwen2.5" in file.lower():
                                        model_id = "bartowski/Qwen2.5-7B-Instruct-GGUF"
                                    else:
                                        model_id = "bartowski/Qwen2-7B-Instruct-GGUF"
                                    break
                    
                    # Si on n'a pas trouvé d'identifiant, essayer de le déduire du nom du fichier
                    if not model_id and "qwen" in file.lower():
                        if "2.5" in file.lower() or "qwen2.5" in file.lower():
                            model_id = "bartowski/Qwen2.5-7B-Instruct-GGUF"
                        else:
                            model_id = "bartowski/Qwen2-7B-Instruct-GGUF"
                    elif not model_id:
                        # Créer un identifiant générique basé sur le chemin
                        # Extraire le dernier répertoire qui contient le fichier
                        path_parts = rel_path.split(os.sep)
                        for part in reversed(path_parts[:-1]):  # Exclure le nom du fichier
                            if part and part != "." and not part.startswith("."):
                                # Essayer de créer un identifiant
                                clean_part = part.replace(" - ", "/").replace("- ", "-").replace(" -", "-").replace(" ", "-").replace("_", "-")
                                if "/" in clean_part:
                                    model_id = clean_part
                                elif len(path_parts) > 1:
                                    # Prendre le répertoire parent comme org
                                    parent_idx = path_parts.index(part) - 1
                                    if parent_idx >= 0:
                                        org = path_parts[parent_idx].replace(" ", "").replace("-", "").replace("_", "")
                                        model_id = f"{org}/{clean_part}"
                                if model_id:
                                    break
                    
                    # Si on a un model_id (trouvé ou déduit), l'ajouter
                    if model_id:
                        if model_id not in all_gguf_files:
                            all_gguf_files[model_id] = []
                        all_gguf_files[model_id].append({
                            "file": file,
                            "path": full_path,
                            "dir": root
                        })
                    else:
                        # Même sans identifiant, on stocke avec un identifiant générique pour ne pas perdre le fichier
                        logger.debug(f"Fichier GGUF trouvé sans identifiant clair: {full_path}")
                        generic_id = f"unknown/{os.path.basename(root)}"
                        if generic_id not in all_gguf_files:
                            all_gguf_files[generic_id] = []
                        all_gguf_files[generic_id].append({
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
    Utilise un scan récursif pour trouver TOUS les fichiers .gguf dans le cache.
    """
    try:
        # Scanner tous les fichiers GGUF récursivement
        gguf_models = _scan_all_gguf_files()
        
        # Aussi scanner via la méthode standard pour complémentarité
        models = _scan_huggingface_models()
        local_models = _scan_local_models_directory()
        all_models = models + local_models
        
        # Ajouter les modèles trouvés par l'ancienne méthode qui ne sont pas déjà dans la liste
        existing_identifiers = {m.get("identifier") for m in gguf_models}
        
        for model in all_models:
            if model.get("model_format") == "gguf" or model.get("gguf_files"):
                model_id = model.get("identifier")
                if model_id and model_id not in existing_identifiers:
                    gguf_info = {
                        "identifier": model_id,
                        "path": model.get("path"),
                        "gguf_files": model.get("gguf_files", []),
                        "recommended_gguf_file": model.get("recommended_gguf_file"),
                        "size_mb": model.get("size_mb", 0),
                        "model_format": "gguf"
                    }
                    
                    # Ajouter le chemin complet du fichier recommandé
                    if gguf_info["recommended_gguf_file"] and gguf_info["path"]:
                        model_dir = Path(gguf_info["path"])
                        if model_dir.exists():
                            full_path = model_dir / gguf_info["recommended_gguf_file"]
                            if full_path.exists():
                                gguf_info["recommended_gguf_file_path"] = str(full_path)
                    
                    gguf_models.append(gguf_info)
                    existing_identifiers.add(model_id)
        
        # Créer une liste simplifiée pour l'affichage
        simplified_models = []
        for model in gguf_models:
            simplified = {
                "identifier": model.get("identifier"),
                "recommended_unique_id": model.get("recommended_gguf_unique_id") or model.get("identifier"),
                "recommended_gguf_file": model.get("recommended_gguf_file"),
                "recommended_gguf_file_path": model.get("recommended_gguf_file_path"),
                "gguf_files": model.get("gguf_files", []),
                "unique_identifiers": model.get("unique_identifiers", []),
                "size_mb": model.get("size_mb", 0),
                "path": model.get("path")
            }
            simplified_models.append(simplified)
        
        return jsonify({
            "status": "success",
            "count": len(gguf_models),
            "cache_dir": str(_get_huggingface_cache_dir()),
            "gguf_models": simplified_models,
            "note": "Utilisez 'identifier' ou 'recommended_unique_id' pour charger un modèle avec POST /models/load/<identifier>"
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des modèles GGUF: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
