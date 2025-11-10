from typing import List, Optional
from src.utils.models.conversation_model import ConversationModel
from src.utils.models.message_model import MessageModel, Role
from src.db.database import Database
import logging

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Repository pour la persistance des conversations et messages.
    Interface pour l'accès aux données (BDD SQLite3) avec pool de connexions.
    """
    
    def __init__(self, database: Optional[Database] = None):
        """
        Initialise le repository avec un pool de connexions.
        
        Args:
            database: Instance de Database (créée par défaut si None)
        """
        self.db = database or Database()
        # Ne pas stocker de connexion directement, utiliser le pool à chaque opération
    
    def create(self, conversation: ConversationModel) -> ConversationModel:
        """
        Crée une nouvelle conversation en BDD.
        
        Args:
            conversation: ConversationModel à sauvegarder
        
        Returns:
            ConversationModel avec l'ID généré
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            try:
                if conversation.id == 0:
                    # Auto-générer l'ID
                    cursor.execute("""
                        INSERT INTO conversations (name, model_name, provider, temperature, message_max, 
                                                 formatted_cache, cache_version, system_prompt)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        conversation.name,
                        conversation.model_name,
                        conversation.provider,
                        conversation.temperature,
                        conversation.message_max,
                        conversation._formatted_cache,
                        conversation._cache_version,
                        conversation.system_prompt
                    ))
                    conversation.id = cursor.lastrowid
                else:
                    # ID fourni explicitement
                    cursor.execute("""
                        INSERT INTO conversations (id, name, model_name, provider, temperature, message_max,
                                                 formatted_cache, cache_version, system_prompt)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        conversation.id,
                        conversation.name,
                        conversation.model_name,
                        conversation.provider,
                        conversation.temperature,
                        conversation.message_max,
                        conversation._formatted_cache,
                        conversation._cache_version,
                        conversation.system_prompt
                    ))
                
                conn.commit()
                logger.debug(f"Conversation {conversation.id} créée")
                
                # Initialiser la liste de messages vide
                conversation.message = []
                return conversation
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Erreur lors de la création de la conversation : {e}")
                raise
    
    def get_by_id(self, id: int) -> Optional[ConversationModel]:
        """
        Récupère une conversation et tous ses messages depuis la BDD.
        
        Args:
            id: ID de la conversation
        
        Returns:
            ConversationModel avec ses messages, ou None si non trouvée
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            # Récupérer la conversation
            cursor.execute("""
                SELECT id, name, model_name, provider, temperature, message_max, 
                       formatted_cache, cache_version, system_prompt
                FROM conversations
                WHERE id = ?
            """, (id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Créer l'objet ConversationModel
            # sqlite3.Row n'a pas de méthode .get(), utiliser l'indexation directe
            provider = row['provider'] if 'provider' in row.keys() else 'local'
            system_prompt = row['system_prompt'] if 'system_prompt' in row.keys() else None
            conversation = ConversationModel(
                id=row['id'],
                name=row['name'],
                model_name=row['model_name'],
                provider=provider,
                temperature=row['temperature'],
                message_max=row['message_max'],
                system_prompt=system_prompt,
                _formatted_cache=row['formatted_cache'],
                _cache_version=row['cache_version']
            )
            
            # Charger les messages associés
            conversation.message = self.get_messages(id)
            
            return conversation
    
    def get_all(self) -> List[ConversationModel]:
        """
        Récupère toutes les conversations depuis la BDD.
        
        Returns:
            Liste de toutes les conversations
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, name, model_name, provider, temperature, message_max,
                       formatted_cache, cache_version, system_prompt
                FROM conversations
                ORDER BY created_at DESC
            """)
            
            conversations = []
            for row in cursor.fetchall():
                # sqlite3.Row n'a pas de méthode .get(), utiliser l'indexation directe
                # Si la colonne provider n'existe pas, utiliser 'local' par défaut
                provider = row['provider'] if 'provider' in row.keys() else 'local'
                system_prompt = row['system_prompt'] if 'system_prompt' in row.keys() else None
                conversation = ConversationModel(
                    id=row['id'],
                    name=row['name'],
                    model_name=row['model_name'],
                    provider=provider,
                    temperature=row['temperature'],
                    message_max=row['message_max'],
                    system_prompt=system_prompt,
                    _formatted_cache=row['formatted_cache'],
                    _cache_version=row['cache_version']
                )
                # Charger les messages (optionnel, peut être fait en lazy loading)
                conversation.message = self.get_messages(conversation.id)
                conversations.append(conversation)
            
            return conversations
    
    def update(self, conversation: ConversationModel) -> ConversationModel:
        """
        Met à jour une conversation existante.
        
        Args:
            conversation: ConversationModel à mettre à jour
        
        Returns:
            ConversationModel mis à jour
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    UPDATE conversations
                    SET name = ?, model_name = ?, provider = ?, temperature = ?, message_max = ?,
                        formatted_cache = ?, cache_version = ?, system_prompt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    conversation.name,
                    conversation.model_name,
                    conversation.provider,
                    conversation.temperature,
                    conversation.message_max,
                    conversation._formatted_cache,
                    conversation._cache_version,
                    conversation.system_prompt,
                    conversation.id
                ))
                
                if cursor.rowcount == 0:
                    raise ValueError(f"Conversation {conversation.id} n'existe pas")
                
                conn.commit()
                logger.debug(f"Conversation {conversation.id} mise à jour")
                return conversation
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Erreur lors de la mise à jour de la conversation : {e}")
                raise
    
    def delete(self, id: int) -> bool:
        """
        Supprime une conversation et tous ses messages.
        Les messages sont supprimés automatiquement grâce à ON DELETE CASCADE.
        
        Args:
            id: ID de la conversation à supprimer
        
        Returns:
            True si supprimée, False si non trouvée
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("DELETE FROM conversations WHERE id = ?", (id,))
                deleted = cursor.rowcount > 0
                
                if deleted:
                    conn.commit()
                    logger.debug(f"Conversation {id} supprimée")
                else:
                    conn.rollback()
                
                return deleted
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Erreur lors de la suppression de la conversation : {e}")
                raise
    
    def delete_all(self) -> int:
        """
        Supprime toutes les conversations et tous leurs messages.
        Les messages sont supprimés automatiquement grâce à ON DELETE CASCADE.
        
        Returns:
            Nombre de conversations supprimées
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("DELETE FROM conversations")
                count = cursor.rowcount
                
                if count > 0:
                    conn.commit()
                    logger.info(f"{count} conversation(s) supprimée(s)")
                else:
                    conn.rollback()
                
                return count
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Erreur lors de la suppression de toutes les conversations : {e}")
                raise
    
    def add_message(self, conversation_id: int, message: MessageModel) -> MessageModel:
        """
        Ajoute un message à une conversation.
        
        Args:
            conversation_id: ID de la conversation
            message: MessageModel à ajouter
        
        Returns:
            MessageModel avec l'ID généré
        
        Raises:
            ValueError: Si la conversation n'existe pas
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            # Vérifier que la conversation existe
            cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            if not cursor.fetchone():
                raise ValueError(f"Conversation {conversation_id} n'existe pas")
            
            try:
                if message.id == 0:
                    # Auto-générer l'ID
                    cursor.execute("""
                        INSERT INTO messages (conversation_id, role, content, temperature)
                        VALUES (?, ?, ?, ?)
                    """, (
                        conversation_id,
                        message.role.value,  # Convertir l'enum en string
                        message.content,
                        message.temperature
                    ))
                    message.id = cursor.lastrowid
                else:
                    # ID fourni explicitement
                    cursor.execute("""
                        INSERT INTO messages (id, conversation_id, role, content, temperature)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        message.id,
                        conversation_id,
                        message.role.value,
                        message.content,
                        message.temperature
                    ))
                
                conn.commit()
                logger.debug(f"Message {message.id} ajouté à la conversation {conversation_id}")
                return message
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Erreur lors de l'ajout du message : {e}")
                raise
    
    def get_messages(self, conversation_id: int) -> List[MessageModel]:
        """
        Récupère tous les messages d'une conversation, triés par date de création.
        
        Args:
            conversation_id: ID de la conversation
        
        Returns:
            Liste des messages
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, conversation_id, role, content, temperature
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            """, (conversation_id,))
            
            messages = []
            for row in cursor.fetchall():
                message = MessageModel(
                    id=row['id'],
                    conversation_id=row['conversation_id'],
                    role=Role(row['role']),  # Convertir le string en enum
                    content=row['content'],
                    temperature=row['temperature']
                )
                messages.append(message)
            
            return messages
    
    def get_message(self, conversation_id: int, message_id: int) -> Optional[MessageModel]:
        """
        Récupère un message d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            message_id: ID du message
        Returns:
            Un message ou None si non trouvé
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, conversation_id, role, content, temperature
                FROM messages
                WHERE id = ? AND conversation_id = ?
            """, (message_id, conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return MessageModel(
                id=row['id'],
                conversation_id=row['conversation_id'],
                role=Role(row['role']),
                content=row['content'],
                temperature=row['temperature']
            )

    def delete_message(self, conversation_id: int, message_id: int) -> bool:
        """
        Supprime un message d'une conversation.
        
        Args:
            conversation_id: ID de la conversation
            message_id: ID du message
        Returns:
            True si supprimé, False si non trouvé
        """
        with self.db.get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    DELETE FROM messages
                    WHERE id = ? AND conversation_id = ?
                """, (message_id, conversation_id))
                
                deleted = cursor.rowcount > 0
                
                if deleted:
                    conn.commit()
                    logger.debug(f"Message {message_id} supprimé de la conversation {conversation_id}")
                else:
                    conn.rollback()
                
                return deleted
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Erreur lors de la suppression du message : {e}")
                raise
