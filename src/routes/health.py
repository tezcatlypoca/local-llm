"""
Routes pour le monitoring et le health check de l'API.
"""
import logging
import sys
from pathlib import Path
from flask import Blueprint, jsonify

# Import depuis le dossier parent (src/)
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from llm_manager_instance import get_llm_manager

logger = logging.getLogger(__name__)

# Création du Blueprint
health_bp = Blueprint('health', __name__)


def _get_gpu_metrics(gpu_id: int):
    """
    Récupère les métriques d'un GPU.
    
    Args:
        gpu_id: ID du GPU (0 ou 1)
    
    Returns:
        Dictionnaire avec les métriques du GPU
    """
    import torch
    
    metrics = {
        "gpu_id": gpu_id,
        "available": False,
        "metrics": {}
    }
    
    if not torch.cuda.is_available():
        return metrics
    
    if gpu_id >= torch.cuda.device_count():
        return metrics
    
    metrics["available"] = True
    
    try:
        device = torch.device(f"cuda:{gpu_id}")
        props = torch.cuda.get_device_properties(gpu_id)
        
        # Mémoire
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)  # GB
        memory_total = props.total_memory / (1024**3)  # GB
        memory_free = memory_total - memory_reserved
        
        metrics["metrics"]["memory"] = {
            "allocated_gb": round(memory_allocated, 2),
            "reserved_gb": round(memory_reserved, 2),
            "total_gb": round(memory_total, 2),
            "free_gb": round(memory_free, 2),
            "utilization_percent": round((memory_reserved / memory_total) * 100, 2) if memory_total > 0 else 0
        }
        
        # Informations du GPU
        metrics["metrics"]["gpu_info"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}" if hasattr(props, 'major') else "unknown",
            "total_memory_gb": round(memory_total, 2)
        }
        
        # Note: PyTorch/ROCm ne fournit pas directement la température ou l'utilisation GPU
        # via l'API Python. Ces informations nécessiteraient des outils système comme
        # rocm-smi, nvidia-smi, ou des bindings spécifiques.
        # On peut suggérer d'utiliser ces outils en externe.
        metrics["metrics"]["note"] = "Pour température et utilisation détaillée, utiliser 'rocm-smi' ou outils système"
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques GPU {gpu_id}: {e}", exc_info=True)
        metrics["error"] = str(e)
    
    return metrics


@health_bp.route('/health', methods=['GET'])
def health_overview():
    """
    Retourne l'état de santé global de l'API et de tous les GPUs.
    
    Returns:
        État global avec informations sur tous les GPUs
    """
    try:
        manager = get_llm_manager()
        status = manager.get_model_status()
        
        # Ajouter des métriques pour chaque GPU
        gpu_health = {}
        for gpu_id in [0, 1]:
            gpu_status = status["gpus"].get(gpu_id, {})
            metrics = _get_gpu_metrics(gpu_id)
            
            gpu_health[gpu_id] = {
                "gpu_id": gpu_id,
                "gpu_identifier": f"GPU-{gpu_id}",
                "available": gpu_status.get("gpu_available", False),
                "model_loaded": gpu_status.get("model_loaded", False),
                "model_name": gpu_status.get("model_name"),
                "device": gpu_status.get("device"),
                "metrics": metrics.get("metrics", {})
            }
            
            # Ajouter les infos mémoire si disponibles
            if "memory" in status["gpus"].get(gpu_id, {}):
                gpu_health[gpu_id]["metrics"]["memory"] = status["gpus"][gpu_id]["memory"]
        
        return jsonify({
            'status': 'healthy',
            'service': 'local-llm-api',
            'total_gpus': status["total_gpus"],
            'gpus': gpu_health,
            'summary': {
                'total_gpus_available': sum(1 for g in gpu_health.values() if g.get("available", False)),
                'gpus_with_models': sum(1 for g in gpu_health.values() if g.get("model_loaded", False)),
                'gpus_free': sum(1 for g in gpu_health.values() if g.get("available", False) and not g.get("model_loaded", False))
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors de la récupération de l\'état de santé: {str(e)}'
        }), 500


@health_bp.route('/health/<int:gpu_id>', methods=['GET'])
def health_gpu(gpu_id: int):
    """
    Retourne l'état de santé détaillé d'un GPU spécifique.
    
    Args:
        gpu_id: Numéro du GPU (0 ou 1)
    
    Returns:
        État détaillé du GPU avec métriques complètes
    """
    try:
        if gpu_id not in [0, 1]:
            return jsonify({
                'status': 'error',
                'message': f'GPU ID invalide: {gpu_id}. Doit être 0 ou 1.'
            }), 400
        
        manager = get_llm_manager()
        gpu_status = manager.get_model_status(gpu_id=gpu_id)
        
        if "error" in gpu_status:
            return jsonify({
                'status': 'error',
                'message': gpu_status["error"]
            }), 400
        
        # Récupérer les métriques détaillées
        metrics = _get_gpu_metrics(gpu_id)
        
        # Construire la réponse complète
        health_data = {
            'status': 'healthy' if gpu_status.get("gpu_available", False) else 'unavailable',
            'gpu_id': gpu_id,
            'gpu_identifier': f'GPU-{gpu_id}',
            'available': gpu_status.get("gpu_available", False),
            'model': {
                'loaded': gpu_status.get("model_loaded", False),
                'name': gpu_status.get("model_name"),
                'has_access_token': gpu_status.get("has_access_token", False)
            },
            'device': gpu_status.get("device"),
            'metrics': metrics.get("metrics", {})
        }
        
        # Ajouter les infos mémoire si disponibles dans gpu_status
        if "memory" in gpu_status:
            health_data["metrics"]["memory"] = gpu_status["memory"]
        
        # Statut plus détaillé
        if not gpu_status.get("gpu_available", False):
            health_data["status"] = "unavailable"
            health_data["message"] = f"GPU {gpu_id} n'est pas disponible"
        elif gpu_status.get("model_loaded", False):
            health_data["status"] = "in_use"
            health_data["message"] = f"GPU {gpu_id} a un modèle chargé: {gpu_status.get('model_name')}"
        else:
            health_data["status"] = "free"
            health_data["message"] = f"GPU {gpu_id} est disponible et libre"
        
        return jsonify(health_data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du health check GPU {gpu_id}: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors de la récupération de l\'état du GPU {gpu_id}: {str(e)}'
        }), 500

