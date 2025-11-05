"""
Routes pour les logs.
"""
from flask import Blueprint, jsonify, request, Response
from ...core.api_client import base_api_client
import logging
import json
import requests

logger = logging.getLogger(__name__)

bp = Blueprint('logs', __name__)


@bp.route('/logs/stream', methods=['GET'])
def stream_logs():
    """
    Stream des logs en temps réel via Server-Sent Events (SSE).
    
    GET /logs/stream?level=INFO&timeout=300
    
    Query parameters:
    - level: Filtrer par niveau de log (optionnel)
    - timeout: Timeout en secondes (optionnel, défaut: 300)
    """
    try:
        level = request.args.get('level')
        timeout = request.args.get('timeout', type=int, default=300)
        
        # Utiliser la méthode stream_logs de LLMClient
        def generate():
            try:
                for log_line in base_api_client.logs.stream_logs(
                    level=level,
                    timeout=timeout
                ):
                    # Retransmettre les données SSE
                    yield f"data: {log_line}\n\n"
            except Exception as e:
                logger.error(f"Erreur lors du streaming des logs: {e}")
                error_data = json.dumps({'type': 'error', 'message': str(e)})
                yield f"data: {error_data}\n\n"
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'  # Pour nginx
            }
        )
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du stream de logs: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la création du stream: {str(e)}"
        }), 500


@bp.route('/logs/history', methods=['GET'])
def logs_history():
    """
    Récupère l'historique des logs récents stockés en mémoire.
    
    GET /logs/history?limit=100&level=INFO&since=1705324245
    
    Query parameters:
    - limit: Nombre maximum de logs à retourner (optionnel, défaut: 100, max: 1000)
    - level: Filtrer par niveau de log (optionnel)
    - since: Timestamp Unix (optionnel)
    """
    try:
        params = {}
        limit = request.args.get('limit', type=int)
        level = request.args.get('level')
        since = request.args.get('since', type=float)
        
        if limit is not None:
            params['limit'] = limit
        if level:
            params['level'] = level
        if since is not None:
            params['since'] = since
        
        result = base_api_client.logs.get_history(
            limit=limit,
            level=level,
            since=since
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique des logs: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la communication avec l'API de base: {str(e)}"
        }), 500


@bp.route('/logs/stats', methods=['GET'])
def logs_stats():
    """
    Retourne des statistiques sur le buffer de logs en mémoire.
    
    GET /logs/stats
    """
    try:
        result = base_api_client.logs.get_stats()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats des logs: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la communication avec l'API de base: {str(e)}"
        }), 500

