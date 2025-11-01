"""
Routes pour consulter les logs en temps réel via API.

Fournit :
- GET /logs/stream : Streaming SSE en temps réel
- GET /logs/history : Historique des logs récents
- GET /logs/stats : Statistiques sur les logs
"""
import json
import logging
import sys
import time
from pathlib import Path
from queue import Empty
from flask import Blueprint, Response, jsonify, request, stream_with_context

# Import depuis le dossier parent (src/)
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.log_buffer import get_log_buffer

logger = logging.getLogger(__name__)

# Création du Blueprint
logs_bp = Blueprint('logs', __name__)


@logs_bp.route('/logs/stream', methods=['GET'])
def stream_logs():
    """
    Stream des logs en temps réel via Server-Sent Events (SSE).
    
    Query parameters:
        - level (optionnel): Filtrer par niveau (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - timeout (optionnel): Timeout en secondes (défaut: 300 = 5 minutes)
    
    Usage:
        Ouvrir cette URL dans un navigateur ou utiliser un client SSE.
        Les logs sont envoyés au format JSON, événement par événement.
    
    Exemple avec curl:
        curl -N http://localhost:5000/logs/stream
    
    Exemple JavaScript:
        const eventSource = new EventSource('/logs/stream?level=INFO');
        eventSource.onmessage = (event) => {
            const log = JSON.parse(event.data);
            console.log(log);
        };
    """
    try:
        log_buffer = get_log_buffer()
        
        # Paramètres de la requête
        level_filter = request.args.get('level', None)
        timeout = int(request.args.get('timeout', 300))  # 5 minutes par défaut
        timeout_time = time.time() + timeout
        
        # Créer une subscription
        sub_id = log_buffer.subscribe()
        queue = log_buffer.get_subscriber_queue(sub_id)
        
        if queue is None:
            return jsonify({
                'status': 'error',
                'message': 'Impossible de créer la subscription'
            }), 500
        
        logger.info(f"Démarrage du streaming SSE (subscription: {sub_id})")
        
        def generate():
            """
            Générateur pour le streaming SSE.
            """
            try:
                # Envoyer un message de démarrage
                start_msg = json.dumps({'type': 'start', 'message': 'Streaming démarré'})
                yield f"data: {start_msg}\n\n"
                
                # Envoyer les logs récents d'abord
                recent_logs = log_buffer.get_logs(limit=50, level=level_filter)
                for log in reversed(recent_logs):  # Plus ancien au plus récent
                    log_json = json.dumps(log)
                    yield f"data: {log_json}\n\n"
                
                # Envoyer un message indiquant le début du streaming en temps réel
                live_msg = json.dumps({'type': 'live', 'message': 'Streaming en temps réel démarré'})
                yield f"data: {live_msg}\n\n"
                
                # Stream en temps réel
                while time.time() < timeout_time:
                    try:
                        # Attendre un nouveau log avec timeout
                        log_entry = queue.get(timeout=1.0)
                        
                        # Filtrer par niveau si demandé
                        if level_filter and log_entry.get('level') != level_filter.upper():
                            continue
                        
                        # Envoyer le log au format SSE
                        log_json = json.dumps(log_entry)
                        yield f"data: {log_json}\n\n"
                        
                    except Empty:
                        # Timeout - envoyer un keepalive pour maintenir la connexion
                        yield f": keepalive\n\n"
                        continue
                    except Exception as e:
                        logger.error(f"Erreur lors du streaming: {e}", exc_info=True)
                        error_msg = json.dumps({'type': 'error', 'message': str(e)})
                        yield f"data: {error_msg}\n\n"
                        break
                
                # Timeout atteint
                timeout_msg = json.dumps({'type': 'timeout', 'message': f'Streaming terminé après {timeout} secondes'})
                yield f"data: {timeout_msg}\n\n"
                
            except GeneratorExit:
                # Le client a fermé la connexion
                logger.info(f"Streaming SSE fermé (subscription: {sub_id})")
            finally:
                # Nettoyer la subscription
                log_buffer.unsubscribe(sub_id)
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Désactiver le buffering nginx si présent
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',  # CORS pour SSE
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    
    except Exception as e:
        logger.error(f"Erreur lors du streaming des logs: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors du streaming: {str(e)}'
        }), 500


@logs_bp.route('/logs/history', methods=['GET'])
def get_logs_history():
    """
    Récupère l'historique des logs récents.
    
    Query parameters:
        - limit (optionnel): Nombre maximum de logs à retourner (défaut: 100, max: 1000)
        - level (optionnel): Filtrer par niveau (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - since (optionnel): Timestamp Unix - retourner uniquement les logs après cette date
    
    Returns:
        Liste des logs (du plus récent au plus ancien)
    """
    try:
        log_buffer = get_log_buffer()
        
        # Paramètres de la requête
        limit = min(int(request.args.get('limit', 100)), 1000)  # Max 1000
        level_filter = request.args.get('level', None)
        since = request.args.get('since', None)
        since_float = float(since) if since else None
        
        # Récupérer les logs
        logs = log_buffer.get_logs(
            limit=limit,
            level=level_filter,
            since=since_float
        )
        
        return jsonify({
            'status': 'success',
            'count': len(logs),
            'logs': logs
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors de la récupération: {str(e)}'
        }), 500


@logs_bp.route('/logs/stats', methods=['GET'])
def get_logs_stats():
    """
    Retourne des statistiques sur les logs.
    
    Returns:
        Statistiques (nombre total, répartition par niveau, etc.)
    """
    try:
        log_buffer = get_log_buffer()
        stats = log_buffer.get_stats()
        
        return jsonify({
            'status': 'success',
            'stats': stats
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors de la récupération: {str(e)}'
        }), 500

