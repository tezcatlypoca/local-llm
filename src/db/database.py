"""
Gestion de la connexion à la base de données SQLite3 et création du schéma.
"""

import sqlite3
import os
import threading
from pathlib import Path
from typing import Optional
import logging
from .connection_pool import ConnectionPool, PooledConnection

logger = logging.getLogger(__name__)

def get_db_path() -> str:
    """
    Retourne le chemin vers le fichier de base de données.
    Par défaut : à la racine du projet dans un dossier 'data/'
    
    Returns:
        Chemin vers le fichier de base de données
    """
    # Chemin à la racine du projet
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    # Créer le dossier data s'il n'existe pas
    data_dir.mkdir(exist_ok=True)
    
    return str(data_dir / "conversations.db")


class Database:
    """
    Gestionnaire de connexion à la base de données SQLite3 avec pool de connexions.
    """
    
    # Pool global partagé (singleton)
    _pool: Optional[ConnectionPool] = None
    _pool_lock = threading.Lock()
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 5,
        enable_wal: bool = True
    ):
        """
        Initialise la connexion à la base de données avec pool.
        
        Args:
            db_path: Chemin vers le fichier de base de données (par défaut: data/conversations.db)
            pool_size: Taille du pool de connexions (défaut: 10)
            max_overflow: Nombre maximum de connexions supplémentaires (défaut: 5)
            enable_wal: Activer le mode WAL pour améliorer la concurrence (défaut: True)
        """
        self.db_path = db_path or get_db_path()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.enable_wal = enable_wal
        
        # Initialiser le pool (singleton par db_path)
        self._init_pool()
        
        # Créer les tables si nécessaire (une seule fois)
        self._ensure_tables_exist()
    
    def _init_pool(self):
        """Initialise le pool de connexions (singleton par db_path)."""
        with Database._pool_lock:
            # Si le pool existe mais avec un autre db_path, le fermer et en créer un nouveau
            if Database._pool is not None and Database._pool.db_path != self.db_path:
                Database._pool.close_all()
                Database._pool = None
            
            if Database._pool is None:
                Database._pool = ConnectionPool(
                    db_path=self.db_path,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    enable_wal=self.enable_wal
                )
            self.pool = Database._pool
    
    def _ensure_tables_exist(self):
        """Crée les tables si elles n'existent pas (une seule fois)."""
        with PooledConnection(self.pool) as conn:
            self._create_tables(conn)
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Obtient une connexion du pool.
        
        ⚠️ ATTENTION : Cette méthode retourne une connexion brute.
        Pour une utilisation automatique avec context manager, utilisez get_pooled_connection().
        
        Returns:
            Connexion SQLite3 du pool
        """
        return self.pool.get_connection()
    
    def return_connection(self, conn: sqlite3.Connection):
        """
        Retourne une connexion au pool.
        
        Args:
            conn: Connexion à retourner
        """
        self.pool.return_connection(conn)
    
    def get_pooled_connection(self) -> PooledConnection:
        """
        Obtient un context manager pour une connexion du pool.
        
        Usage:
            with db.get_pooled_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM ...")
        
        Returns:
            PooledConnection (context manager)
        """
        return PooledConnection(self.pool)
    
    def _create_tables(self, conn: sqlite3.Connection):
        """
        Crée les tables de la base de données si elles n'existent pas.
        
        Args:
            conn: Connexion SQLite3
        """
        cursor = conn.cursor()
        
        # Table conversations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT 'local',
                temperature REAL NOT NULL DEFAULT 0.7,
                message_max INTEGER NOT NULL DEFAULT 0,
                formatted_cache TEXT,
                cache_version INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Migrations : Ajouter les colonnes si elles n'existent pas (pour les bases existantes)
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'provider' not in columns:
            try:
                cursor.execute("ALTER TABLE conversations ADD COLUMN provider TEXT NOT NULL DEFAULT 'local'")
                conn.commit()
                logger.debug("Colonne 'provider' ajoutée à la table conversations")
            except sqlite3.OperationalError as e:
                logger.warning(f"Impossible d'ajouter la colonne 'provider': {e}")
        
        if 'system_prompt' not in columns:
            try:
                cursor.execute("ALTER TABLE conversations ADD COLUMN system_prompt TEXT")
                conn.commit()
                logger.debug("Colonne 'system_prompt' ajoutée à la table conversations")
            except sqlite3.OperationalError as e:
                logger.warning(f"Impossible d'ajouter la colonne 'system_prompt': {e}")
        
        # Table messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                temperature REAL NOT NULL DEFAULT 0.7,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        
        # Index pour améliorer les performances
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
            ON messages(conversation_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_created_at 
            ON messages(created_at)
        """)
        
        conn.commit()
        logger.debug("Tables de base de données créées/vérifiées")
    
    def close(self):
        """
        Ferme toutes les connexions du pool.
        ⚠️ À utiliser uniquement lors de l'arrêt de l'application.
        """
        if Database._pool:
            Database._pool.close_all()
            with Database._pool_lock:
                Database._pool = None
            logger.debug("Pool de connexions fermé")
    
    def get_stats(self) -> dict:
        """
        Retourne les statistiques du pool.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        if self.pool:
            return self.pool.get_stats()
        return {}
    
    def __enter__(self):
        """Support du context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support du context manager."""
        # Ne pas fermer le pool automatiquement (singleton partagé)
        pass

