"""
Endpoint pour consulter les logs de l'API.
"""
from typing import Dict, Any, Optional, Iterator
from .base import BaseEndpoint
import requests


class LogsEndpoint(BaseEndpoint):
    """
    Endpoint pour consulter les logs de l'API.
    
    Permet de récupérer l'historique des logs, les statistiques,
    et de streamer les logs en temps réel.
    """
    
    def get_history(
        self,
        limit: Optional[int] = None,
        level: Optional[str] = None,
        since: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Récupère l'historique des logs récents.
        
        Args:
            limit: Nombre maximum de logs à retourner (défaut: 100, max: 1000)
            level: Filtrer par niveau de log 
                   ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            since: Timestamp Unix - retourner uniquement les logs après cette date
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "success",
                "count": int,
                "logs": List[Dict]  # Liste des logs avec leurs détails
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur
            requests.RequestException: Si la requête échoue
        """
        params = {}
        if limit is not None:
            params["limit"] = limit
        if level is not None:
            params["level"] = level
        if since is not None:
            params["since"] = since
        
        return self.get("/logs/history", params=params if params else None)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le buffer de logs en mémoire.
        
        Returns:
            Dictionnaire contenant:
            {
                "status": "success",
                "stats": {
                    "total_logs": int,
                    "max_size": int,
                    "level_counts": {
                        "INFO": int,
                        "WARNING": int,
                        "ERROR": int,
                        "DEBUG": int,
                        "CRITICAL": int
                    },
                    "active_subscribers": int
                }
            }
        
        Raises:
            requests.HTTPError: Si l'API retourne une erreur
            requests.RequestException: Si la requête échoue
        """
        return self.get("/logs/stats")
    
    def stream_logs(
        self,
        level: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Iterator[str]:
        """
        Stream des logs en temps réel via Server-Sent Events (SSE).
        
        Cette méthode retourne un générateur qui yield les logs au fur et à mesure
        qu'ils arrivent.
        
        Args:
            level: Filtrer par niveau de log 
                   ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            timeout: Timeout en secondes (défaut: 300, max recommandé: 3600)
        
        Yields:
            Chaînes JSON représentant chaque log reçu
        
        Raises:
            requests.RequestException: Si la connexion échoue
        
        Example:
            ```python
            for log_json in client.logs.stream_logs(level="INFO"):
                log = json.loads(log_json)
                print(log["message"])
            ```
        """
        params = {}
        if level is not None:
            params["level"] = level
        if timeout is not None:
            params["timeout"] = timeout
        
        url = self._build_url("/logs/stream")
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        try:
            response = self.session.get(
                url,
                stream=True,
                timeout=self.client.timeout if timeout is None else timeout + 10,
                headers={"Accept": "text/event-stream"}
            )
            response.raise_for_status()
            
            # Parser le stream SSE
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        # Format SSE: "data: {...}"
                        yield line[6:]  # Retirer le préfixe "data: "
                    elif line.startswith("{"):
                        # Format JSON direct (sans préfixe SSE)
                        yield line
        except requests.RequestException as e:
            raise requests.RequestException(
                f"Erreur lors du streaming des logs: {e}"
            ) from e

