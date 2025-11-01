"""
Module pour gérer un buffer de logs en mémoire avec support pour SSE (Server-Sent Events).

Ce module fournit :
- Un handler de logging personnalisé qui capture les logs dans un buffer circulaire
- Un système de subscriptions pour le streaming en temps réel via SSE
"""
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from queue import Queue, Empty


class LogBuffer:
    """
    Buffer circulaire pour stocker les logs avec support pour SSE streaming.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialise le buffer de logs.
        
        Args:
            max_size: Nombre maximum de logs à conserver en mémoire (défaut: 1000)
        """
        self.max_size = max_size
        self.logs: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        # Dictionnaire pour stocker les queues de subscribers (pour SSE)
        self.subscribers: Dict[str, Queue] = {}
        self.next_subscriber_id = 0
        
    def add_log(self, record: logging.LogRecord):
        """
        Ajoute un log au buffer et notifie tous les subscribers.
        
        Args:
            record: Le LogRecord du logging Python
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
            'lineno': record.lineno
        }
        
        # Ajouter les informations d'exception si présentes
        if record.exc_info:
            import traceback
            log_entry['exception'] = traceback.format_exception(*record.exc_info)
        
        with self.lock:
            self.logs.append(log_entry)
            
            # Notifier tous les subscribers
            subscribers_to_remove = []
            for sub_id, queue in self.subscribers.items():
                try:
                    queue.put_nowait(log_entry)
                except:
                    # Queue pleine ou subscriber mort - le retirer
                    subscribers_to_remove.append(sub_id)
            
            # Nettoyer les subscribers morts
            for sub_id in subscribers_to_remove:
                self.subscribers.pop(sub_id, None)
    
    def get_logs(
        self, 
        limit: Optional[int] = None, 
        level: Optional[str] = None,
        since: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les logs du buffer avec filtres optionnels.
        
        Args:
            limit: Nombre maximum de logs à retourner (None = tous)
            level: Filtrer par niveau (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            since: Timestamp Unix - retourner uniquement les logs après cette date
        
        Returns:
            Liste des logs (du plus récent au plus ancien)
        """
        with self.lock:
            logs = list(self.logs)
        
        # Appliquer les filtres
        if level:
            logs = [log for log in logs if log['level'] == level.upper()]
        
        if since:
            logs = [
                log for log in logs 
                if datetime.fromisoformat(log['timestamp']).timestamp() >= since
            ]
        
        # Limiter le nombre de résultats
        if limit:
            logs = logs[-limit:] if limit > 0 else logs
        
        # Retourner du plus récent au plus ancien
        return list(reversed(logs))
    
    def subscribe(self) -> str:
        """
        Crée une nouvelle subscription pour recevoir les logs en temps réel.
        
        Returns:
            ID unique de la subscription
        """
        sub_id = f"sub_{self.next_subscriber_id}_{int(time.time())}"
        self.next_subscriber_id += 1
        
        with self.lock:
            # Queue avec une taille limitée pour éviter l'accumulation
            self.subscribers[sub_id] = Queue(maxsize=100)
        
        return sub_id
    
    def unsubscribe(self, sub_id: str):
        """Retire une subscription."""
        with self.lock:
            self.subscribers.pop(sub_id, None)
    
    def get_subscriber_queue(self, sub_id: str) -> Optional[Queue]:
        """
        Récupère la queue d'un subscriber.
        
        Args:
            sub_id: ID de la subscription
        
        Returns:
            Queue du subscriber ou None si n'existe pas
        """
        with self.lock:
            return self.subscribers.get(sub_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le buffer.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        with self.lock:
            total_logs = len(self.logs)
            level_counts = {}
            for log in self.logs:
                level = log['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            return {
                'total_logs': total_logs,
                'max_size': self.max_size,
                'level_counts': level_counts,
                'active_subscribers': len(self.subscribers)
            }


class LogBufferHandler(logging.Handler):
    """
    Handler de logging personnalisé qui envoie les logs vers un LogBuffer.
    """
    
    def __init__(self, log_buffer: LogBuffer):
        """
        Initialise le handler.
        
        Args:
            log_buffer: Instance du LogBuffer où stocker les logs
        """
        super().__init__()
        self.log_buffer = log_buffer
    
    def emit(self, record: logging.LogRecord):
        """
        Émet un log vers le buffer.
        
        Args:
            record: Le LogRecord à émettre
        """
        try:
            self.log_buffer.add_log(record)
        except Exception:
            # Ignorer les erreurs pour éviter les boucles infinies de logging
            self.handleError(record)


# Instance globale du buffer de logs
_log_buffer_instance: Optional[LogBuffer] = None
_log_buffer_lock = threading.Lock()


def get_log_buffer() -> LogBuffer:
    """
    Retourne l'instance globale du buffer de logs (singleton).
    
    Returns:
        Instance unique du LogBuffer
    """
    global _log_buffer_instance
    
    with _log_buffer_lock:
        if _log_buffer_instance is None:
            _log_buffer_instance = LogBuffer(max_size=1000)
        return _log_buffer_instance


def setup_log_buffer_handler(logger_name: Optional[str] = None):
    """
    Configure le handler de buffer de logs pour un logger spécifique ou le root logger.
    
    Args:
        logger_name: Nom du logger (None = root logger)
    
    Returns:
        Le handler créé
    """
    log_buffer = get_log_buffer()
    handler = LogBufferHandler(log_buffer)
    
    # Formatter simple
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)  # Capturer tous les niveaux
    
    # Ajouter au logger
    logger = logging.getLogger(logger_name) if logger_name else logging.root
    logger.addHandler(handler)
    
    return handler

