"""
Implémentation SQLite pour le stockage des conversations.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from ..context import ConversationContext, Message, MessageRole
from ..exceptions import ConversationNotFoundError
from .base import StorageBackend


class SQLiteStorage(StorageBackend):
    """
    Stockage SQLite pour les conversations.
    
    Crée automatiquement la base de données et les tables
    si elles n'existent pas.
    """
    
    def __init__(self, db_path: str = "conversations.db"):
        """
        Initialise le stockage SQLite.
        
        Args:
            db_path: Chemin vers le fichier de base de données
        """
        self.db_path = Path(db_path)
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Crée une connexion à la base de données."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,  # Timeout pour les accès concurrents
            check_same_thread=False  # Permet l'usage multi-thread si nécessaire
        )
        # Activer les clés étrangères
        conn.execute("PRAGMA foreign_keys = ON")
        # Mode WAL pour meilleures performances
        conn.execute("PRAGMA journal_mode = WAL")
        # Synchronisation normale (bon compromis performance/sécurité)
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn
    
    def _init_database(self):
        """Crée les tables si elles n'existent pas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table des conversations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    gpu_id INTEGER NOT NULL,
                    system_prompt TEXT,
                    max_messages INTEGER,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Table des messages
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Index pour améliorer les performances
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_model 
                ON conversations(model_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated 
                ON conversations(updated_at)
            """)
            
            conn.commit()
    
    def save_conversation(self, context: ConversationContext):
        """Sauvegarde une conversation complète."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Convertir les données en format JSON pour la sérialisation
            metadata_json = json.dumps({})  # Pour l'instant vide, peut être étendu
            
            # Insérer ou mettre à jour la conversation
            cursor.execute("""
                INSERT OR REPLACE INTO conversations 
                (id, model_name, gpu_id, system_prompt, max_messages, summary, 
                 created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                context.conversation_id,
                context.model_name,
                context.gpu_id,
                context.system_prompt,
                context.max_messages,
                context.summary,
                context.created_at.isoformat(),
                context.updated_at.isoformat(),
                metadata_json
            ))
            
            # Supprimer les anciens messages (on va les réinsérer)
            cursor.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (context.conversation_id,)
            )
            
            # Insérer tous les messages
            for msg in context.messages:
                cursor.execute("""
                    INSERT INTO messages 
                    (conversation_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    context.conversation_id,
                    msg.role.value,
                    msg.content,
                    msg.timestamp.isoformat(),
                    json.dumps(msg.metadata)
                ))
            
            conn.commit()
    
    def load_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Charge une conversation par son ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Charger les métadonnées de la conversation
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Récupérer les noms de colonnes
            column_names = [description[0] for description in cursor.description]
            conv_data = dict(zip(column_names, row))
            
            # Charger les messages
            cursor.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
                (conversation_id,)
            )
            message_rows = cursor.fetchall()
            
            # Récupérer les noms de colonnes pour les messages
            msg_column_names = [description[0] for description in cursor.description]
            
            # Reconstruire le contexte
            context = ConversationContext(
                conversation_id=conv_data["id"],
                model_name=conv_data["model_name"],
                gpu_id=conv_data["gpu_id"],
                max_messages=conv_data["max_messages"],
                system_prompt=conv_data["system_prompt"]
            )
            
            context.summary = conv_data.get("summary")
            context.created_at = datetime.fromisoformat(conv_data["created_at"])
            context.updated_at = datetime.fromisoformat(conv_data["updated_at"])
            
            # Reconstruire les messages
            for msg_row in message_rows:
                msg_data = dict(zip(msg_column_names, msg_row))
                message = Message(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    metadata=json.loads(msg_data["metadata"] or "{}")
                )
                context.messages.append(message)
            
            return context
    
    def list_conversations(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """Liste les IDs des conversations (optionnellement filtrées)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT id FROM conversations"
            params = []
            
            if model_name:
                query += " WHERE model_name = ?"
                params.append(model_name)
            
            query += " ORDER BY updated_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [row[0] for row in rows]
    
    def delete_conversation(self, conversation_id: str):
        """Supprime une conversation et tous ses messages (CASCADE)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            
            # Les messages sont supprimés automatiquement grâce à ON DELETE CASCADE
            # Mais on peut aussi les supprimer explicitement pour être sûr
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            
            conn.commit()
    
    def add_message(self, conversation_id: str, message: Message):
        """Ajoute un message à une conversation existante."""
        # Vérifier que la conversation existe
        context = self.load_conversation(conversation_id)
        if context is None:
            raise ConversationNotFoundError(conversation_id)
        
        # Ajouter le message au contexte
        context.add_message(message.role, message.content, message.metadata)
        
        # Sauvegarder le contexte mis à jour
        self.save_conversation(context)

