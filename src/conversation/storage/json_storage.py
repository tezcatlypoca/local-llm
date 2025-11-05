"""
Implémentation JSON pour le stockage des conversations.
Stocke les données dans des fichiers JSON dans le dossier src/data/.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from ..context import ConversationContext, Message, MessageRole
from ..exceptions import ConversationNotFoundError
from .base import StorageBackend


class JSONStorage(StorageBackend):
    """
    Stockage JSON pour les conversations.
    
    Chaque conversation est stockée dans un fichier JSON séparé
    dans le dossier src/data/conversations/.
    
    Structure:
    - src/data/conversations/{conversation_id}.json
    - src/data/conversations/_index.json (liste des conversations)
    """
    
    def __init__(self, data_dir: str = "src/data"):
        """
        Initialise le stockage JSON.
        
        Args:
            data_dir: Dossier de base pour stocker les données
        """
        self.data_dir = Path(data_dir)
        self.conversations_dir = self.data_dir / "conversations"
        self.index_file = self.conversations_dir / "_index.json"
        
        # Créer les dossiers si nécessaire
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser l'index si nécessaire
        self._init_index()
    
    def _init_index(self):
        """Initialise le fichier d'index s'il n'existe pas."""
        if not self.index_file.exists():
            self._write_index({})
    
    def _read_index(self) -> dict:
        """Lit le fichier d'index."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            # Si erreur de lecture, retourner un index vide
            return {}
    
    def _write_index(self, index: dict):
        """Écrit le fichier d'index."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"Erreur lors de l'écriture de l'index: {e}")
    
    def _get_conversation_file(self, conversation_id: str) -> Path:
        """Retourne le chemin du fichier JSON pour une conversation."""
        return self.conversations_dir / f"{conversation_id}.json"
    
    def save_conversation(self, context: ConversationContext):
        """Sauvegarde une conversation dans un fichier JSON."""
        try:
            # Convertir le contexte en dictionnaire
            data = context.to_dict()
            
            # Sauvegarder dans le fichier
            conversation_file = self._get_conversation_file(context.conversation_id)
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Mettre à jour l'index
            index = self._read_index()
            index[context.conversation_id] = {
                "model_name": context.model_name,
                "gpu_id": context.gpu_id,
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
                "message_count": context.get_message_count()
            }
            self._write_index(index)
            
        except Exception as e:
            raise Exception(f"Erreur lors de la sauvegarde de la conversation: {e}")
    
    def load_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Charge une conversation depuis un fichier JSON."""
        try:
            conversation_file = self._get_conversation_file(conversation_id)
            
            if not conversation_file.exists():
                return None
            
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruire le contexte
            return ConversationContext.from_dict(data)
            
        except Exception as e:
            # En cas d'erreur, retourner None
            return None
    
    def list_conversations(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """Liste les IDs des conversations depuis l'index."""
        try:
            index = self._read_index()
            
            # Filtrer par modèle si nécessaire
            if model_name:
                conversation_ids = [
                    conv_id for conv_id, info in index.items()
                    if info.get("model_name") == model_name
                ]
            else:
                conversation_ids = list(index.keys())
            
            # Trier par updated_at (le plus récent en premier)
            if conversation_ids:
                # Charger les conversations pour trier (peut être optimisé)
                conversations = []
                for conv_id in conversation_ids:
                    conv = self.load_conversation(conv_id)
                    if conv:
                        conversations.append((conv_id, conv.updated_at))
                
                conversations.sort(key=lambda x: x[1], reverse=True)
                conversation_ids = [conv_id for conv_id, _ in conversations]
            
            # Appliquer la limite
            if limit:
                conversation_ids = conversation_ids[:limit]
            
            return conversation_ids
            
        except Exception as e:
            # En cas d'erreur, retourner une liste vide
            return []
    
    def delete_conversation(self, conversation_id: str):
        """Supprime une conversation et son fichier JSON."""
        try:
            conversation_file = self._get_conversation_file(conversation_id)
            
            if not conversation_file.exists():
                raise ConversationNotFoundError(conversation_id)
            
            # Supprimer le fichier
            conversation_file.unlink()
            
            # Mettre à jour l'index
            index = self._read_index()
            if conversation_id in index:
                del index[conversation_id]
                self._write_index(index)
            
        except ConversationNotFoundError:
            raise
        except Exception as e:
            raise Exception(f"Erreur lors de la suppression de la conversation: {e}")
    
    def add_message(self, conversation_id: str, message: Message):
        """Ajoute un message à une conversation existante."""
        # Charger la conversation
        context = self.load_conversation(conversation_id)
        if context is None:
            raise ConversationNotFoundError(conversation_id)
        
        # Ajouter le message
        context.add_message(message.role, message.content, message.metadata)
        
        # Sauvegarder
        self.save_conversation(context)

