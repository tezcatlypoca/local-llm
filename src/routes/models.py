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
    # Par défaut, Hugging Face utilise ~/.cache/huggingface/hub
    # Cette méthode fonctionne sur Windows et Linux
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    return cache_dir


def _scan_huggingface_models():
    """Scanne le cache Hugging Face pour trouver les modèles téléchargés."""
    models = []
    
    # Méthode 1: Utiliser l'API Hugging Face Hub (plus fiable)
    if HF_HUB_AVAILABLE:
        try:
            cache_info = scan_cache_dir()
            # cache_info.repos est un frozenset, on le convertit en liste
            for repo in list(cache_info.repos):
                # repo.revisions peut être un frozenset, on le convertit en liste
                revisions_list = list(repo.revisions) if repo.revisions else []
                if not revisions_list:
                    continue
                
                # Prendre la dernière révision
                latest_revision = revisions_list[-1]
                
                # Calculer la taille (size_on_disk peut être un int en bytes ou un str)
                size_bytes = 0
                if hasattr(latest_revision, 'size_on_disk'):
                    size_val = latest_revision.size_on_disk
                    if isinstance(size_val, (int, float)):
                        size_bytes = size_val
                    elif isinstance(size_val, str):
                        # Si c'est une string, essayer de parser (ex: "500MB")
                        try:
                            size_bytes = int(size_val)
                        except (ValueError, TypeError):
                            size_bytes = 0
                
                # Extraire les noms de fichiers de manière robuste
                file_names = []
                if hasattr(latest_revision, 'files') and latest_revision.files:
                    for f in list(latest_revision.files):
                        # CachedFileInfo peut avoir différents attributs selon la version
                        if hasattr(f, 'file_path'):
                            file_names.append(os.path.basename(f.file_path))
                        elif hasattr(f, 'blob_path'):
                            file_names.append(os.path.basename(f.blob_path))
                        elif hasattr(f, 'filename'):
                            file_names.append(f.filename)
                        elif hasattr(f, 'file_name'):
                            file_names.append(f.file_name)
                        elif hasattr(f, 'path'):
                            file_names.append(os.path.basename(f.path))
                
                model_info = {
                    "identifier": repo.repo_id,
                    "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes > 0 else 0,
                    "revision": latest_revision.commit_hash if hasattr(latest_revision, 'commit_hash') else None,
                    "path": latest_revision.snapshot_path if hasattr(latest_revision, 'snapshot_path') else None,
                    "files": file_names,
                }
                
                # Chercher config.json pour plus d'infos
                if model_info["path"]:
                    config_path = os.path.join(model_info["path"], "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
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
    
    # Méthode 2: Scan manuel (fallback)
    cache_dir = _get_huggingface_cache_dir()
    
    if not os.path.exists(cache_dir):
        logger.debug(f"Cache Hugging Face introuvable: {cache_dir}")
        return models
    
    try:
        # Le cache Hugging Face contient des dossiers avec des identifiants
        # Format: models--{org}--{model_name}--{hash} ou snapshots--{hash}
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue
            
            # Chercher des fichiers de modèle typiques
            has_model_files = False
            config_path = None
            model_info = {
                "path": item_path,
                "files": []
            }
            
            # Parcourir récursivement pour trouver les fichiers de modèle
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    # Fichiers de modèle typiques
                    if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.gguf')):
                        has_model_files = True
                        rel_path = os.path.relpath(os.path.join(root, file), item_path)
                        model_info["files"].append(rel_path)
                    # Fichier de config
                    elif file == 'config.json':
                        config_path = os.path.join(root, file)
                    # Tokenizer
                    elif file in ('tokenizer.json', 'vocab.json', 'merges.txt'):
                        rel_path = os.path.relpath(os.path.join(root, file), item_path)
                        model_info.setdefault("tokenizer_files", []).append(rel_path)
            
            if has_model_files:
                # Lire les infos du modèle depuis config.json si disponible
                if config_path and os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            model_info["model_type"] = config.get("model_type", "unknown")
                            model_info["architectures"] = config.get("architectures", [])
                    except Exception as e:
                        logger.debug(f"Erreur lecture config.json: {e}")
                
                # Essayer d'extraire un nom de modèle depuis le chemin
                # Les dossiers Hugging Face ont souvent le format: models--{org}--{model_name}--{hash}
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
        # Chercher un dossier 'models' dans le projet
        project_root = Path(__file__).parent.parent.parent
        local_dir = project_root / "models"
    
    if isinstance(local_dir, str):
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
                
                # Chercher les fichiers de modèle
                for root, dirs, files in os.walk(item):
                    for file in files:
                        if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.gguf')):
                            model_info["files"].append(os.path.relpath(os.path.join(root, file), item))
                        elif file == 'config.json':
                            config_path = os.path.join(root, file)
                            try:
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    config = json.load(f)
                                    model_info["model_type"] = config.get("model_type", "unknown")
                                    model_info["architectures"] = config.get("architectures", [])
                            except Exception as e:
                                logger.debug(f"Erreur lecture config.json: {e}")
                
                if model_info["files"]:
                    model_info["size_mb"] = _calculate_directory_size(item) / (1024 * 1024)
                    models.append(model_info)
    
    except Exception as e:
        logger.error(f"Erreur lors du scan du dossier local: {e}", exc_info=True)
    
    return models


def _calculate_directory_size(directory):
    """Calcule la taille totale d'un répertoire en octets."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logger.debug(f"Erreur calcul taille: {e}")
    return total_size


@models_bp.route('/models', methods=['GET'])
def list_models():
    """
    Liste tous les modèles disponibles en local.
    
    Retourne:
        - Les modèles du cache Hugging Face (Windows et Linux)
        - Les modèles dans un dossier local (si configuré)
    
    Cette route fonctionne sur Windows et Linux car elle utilise
    os.path.expanduser("~/.cache/huggingface/hub") qui est portable.
    """
    try:
        models = []
        
        # Scanner le cache Hugging Face
        hf_models = _scan_huggingface_models()
        models.extend(hf_models)
        
        # Scanner un dossier local (optionnel)
        local_models = _scan_local_models_directory()
        models.extend(local_models)
        
        # Informations de debug
        cache_dir = _get_huggingface_cache_dir()
        
        return jsonify({
            'status': 'success',
            'count': len(models),
            'cache_dir': cache_dir,
            'cache_exists': os.path.exists(cache_dir),
            'models': models
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la liste des modèles: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

