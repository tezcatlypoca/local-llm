"""
Pool de connexions SQLite3 thread-safe.
"""

import sqlite3
import threading
from queue import Queue, Empty
from typing import Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Pool de connexions SQLite3 thread-safe.
    
    Gère un pool de connexions réutilisables pour améliorer les performances
    et la gestion de la concurrence.
    """
    
    def __init__(
        self,
        db_path: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        timeout: float = 5.0,
        enable_wal: bool = True
    ):
        """
        Initialise le pool de connexions.
        
        Args:
            db_path: Chemin vers le fichier de base de données
            pool_size: Nombre de connexions dans le pool (défaut: 10)
            max_overflow: Nombre maximum de connexions supplémentaires (défaut: 5)
            timeout: Timeout en secondes pour obtenir une connexion (défaut: 5.0)
            enable_wal: Activer le mode WAL pour améliorer la concurrence (défaut: True)
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.enable_wal = enable_wal
        
        # Queue thread-safe pour stocker les connexions disponibles
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created_connections = 0
        self._active_connections = 0
        
        # Initialiser le pool avec des connexions
        self._initialize_pool()
        
        logger.info(
            f"Pool de connexions initialisé : {pool_size} connexions, "
            f"max_overflow={max_overflow}, WAL={'activé' if enable_wal else 'désactivé'}"
        )
    
    def _create_connection(self) -> sqlite3.Connection:
        """
        Crée une nouvelle connexion SQLite3 configurée.
        
        Returns:
            Connexion SQLite3 configurée
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Permet l'utilisation multi-thread
            timeout=20.0  # Timeout pour les opérations de verrouillage
        )
        
        # Activer les clés étrangères
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Activer le mode WAL pour améliorer la concurrence
        # WAL permet : lectures parallèles illimitées + 1 écriture à la fois
        if self.enable_wal:
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                logger.debug("Mode WAL activé pour la connexion")
            except sqlite3.Error as e:
                logger.warning(f"Impossible d'activer le mode WAL: {e}")
        
        # Optimisations de performance
        conn.execute("PRAGMA synchronous = NORMAL")  # Équilibre performance/sécurité
        conn.execute("PRAGMA cache_size = -64000")  # 64MB de cache
        conn.execute("PRAGMA temp_store = MEMORY")  # Tables temporaires en mémoire
        
        # Utiliser Row pour accéder aux colonnes par nom
        conn.row_factory = sqlite3.Row
        
        return conn
    
    def _initialize_pool(self):
        """Initialise le pool avec des connexions."""
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                self._pool.put(conn)
                self._created_connections += 1
            except sqlite3.Error as e:
                logger.error(f"Erreur lors de la création d'une connexion: {e}")
                raise
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Obtient une connexion du pool.
        
        Returns:
            Connexion SQLite3
            
        Raises:
            TimeoutError: Si aucune connexion n'est disponible dans le timeout
        """
        try:
            # Essayer d'obtenir une connexion du pool
            conn = self._pool.get(timeout=self.timeout)
            
            # Vérifier que la connexion est toujours valide
            try:
                conn.execute("SELECT 1").fetchone()
            except sqlite3.Error:
                # Connexion invalide, en créer une nouvelle
                logger.warning("Connexion invalide détectée, création d'une nouvelle")
                conn.close()
                conn = self._create_connection()
                with self._lock:
                    self._created_connections += 1
            
            with self._lock:
                self._active_connections += 1
            
            return conn
            
        except Empty:
            # Pool vide, vérifier si on peut créer une connexion supplémentaire
            with self._lock:
                if self._created_connections < self.pool_size + self.max_overflow:
                    logger.debug("Création d'une connexion supplémentaire (overflow)")
                    conn = self._create_connection()
                    self._created_connections += 1
                    self._active_connections += 1
                    return conn
            
            # Aucune connexion disponible
            raise TimeoutError(
                f"Aucune connexion disponible dans le pool après {self.timeout}s. "
                f"Connexions actives: {self._active_connections}/{self.pool_size + self.max_overflow}"
            )
    
    def return_connection(self, conn: sqlite3.Connection):
        """
        Retourne une connexion au pool.
        
        Args:
            conn: Connexion à retourner
        """
        if conn is None:
            return
        
        try:
            # Vérifier que la connexion est toujours valide
            conn.execute("SELECT 1").fetchone()
            
            # Annuler toute transaction en cours
            conn.rollback()
            
            # Retourner au pool si il y a de la place
            try:
                self._pool.put_nowait(conn)
            except:
                # Pool plein, fermer la connexion
                logger.debug("Pool plein, fermeture de la connexion supplémentaire")
                conn.close()
                with self._lock:
                    self._created_connections -= 1
            
            with self._lock:
                self._active_connections -= 1
                
        except sqlite3.Error as e:
            # Connexion invalide, la fermer
            logger.warning(f"Connexion invalide lors du retour au pool: {e}")
            try:
                conn.close()
            except:
                pass
            with self._lock:
                self._created_connections -= 1
                self._active_connections -= 1
    
    def close_all(self):
        """Ferme toutes les connexions du pool."""
        logger.info("Fermeture de toutes les connexions du pool")
        
        # Fermer les connexions dans le pool
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
        
        with self._lock:
            self._created_connections = 0
            self._active_connections = 0
    
    def get_stats(self) -> dict:
        """
        Retourne les statistiques du pool.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "created_connections": self._created_connections,
                "active_connections": self._active_connections,
                "available_connections": self._pool.qsize(),
                "wal_enabled": self.enable_wal
            }
    
    def __enter__(self):
        """Support du context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support du context manager."""
        self.close_all()


class PooledConnection:
    """
    Wrapper pour une connexion du pool avec support du context manager.
    """
    
    def __init__(self, pool: ConnectionPool):
        """
        Initialise le wrapper.
        
        Args:
            pool: Pool de connexions
        """
        self.pool = pool
        self.conn: Optional[sqlite3.Connection] = None
    
    def __enter__(self) -> sqlite3.Connection:
        """Obtient une connexion du pool."""
        self.conn = self.pool.get_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Retourne la connexion au pool."""
        if self.conn:
            self.pool.return_connection(self.conn)
            self.conn = None

