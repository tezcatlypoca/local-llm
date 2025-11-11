"""
Module RAG (Retrieval-Augmented Generation) générique.

Ce module fournit une classe RAGManager pour gérer un système RAG complet :
- Chargement et parsing de documents
- Découpage en chunks
- Génération d'embeddings
- Stockage dans une base de données vectorielle
- Recherche de documents pertinents
"""

import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    Settings = None
    embedding_functions = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingFunction:
    """
    Classe wrapper pour utiliser SentenceTransformer avec ChromaDB.
    Compatible avec ChromaDB 0.4.16+ qui attend une signature avec 'input'.
    """
    
    def __init__(self, model: SentenceTransformer):
        """
        Initialise la fonction d'embedding.
        
        Args:
            model: Instance de SentenceTransformer
        """
        self.model = model
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de textes.
        
        Args:
            input: Liste de textes à encoder
            
        Returns:
            Liste de listes de floats (embeddings)
        """
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class DocumentLoader:
    """Classe pour charger et parser différents types de documents."""
    
    # Formats supportés
    SUPPORTED_FORMATS = {'.txt', '.md', '.pdf', '.docx'}
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Vérifie si le format du fichier est supporté."""
        extension = Path(file_path).suffix.lower()
        return extension in DocumentLoader.SUPPORTED_FORMATS
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Retourne la liste des formats supportés."""
        return sorted(list(DocumentLoader.SUPPORTED_FORMATS))
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Charge un fichier texte."""
        file_path_obj = Path(file_path)
        
        # Vérifier que le fichier existe
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas: {file_path}")
        
        # Vérifier que c'est un fichier (pas un répertoire)
        if not file_path_obj.is_file():
            raise ValueError(f"Le chemin spécifié n'est pas un fichier: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"Le fichier {file_path} est vide")
                return content
        except UnicodeDecodeError as e:
            logger.error(f"Erreur d'encodage lors du chargement de {file_path}: {e}")
            raise ValueError(f"Impossible de décoder le fichier {file_path} en UTF-8. Le fichier pourrait être binaire ou utiliser un autre encodage.")
        except PermissionError as e:
            logger.error(f"Permission refusée pour {file_path}: {e}")
            raise PermissionError(f"Permission refusée pour accéder au fichier: {file_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier {file_path}: {e}")
            raise
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Charge un fichier PDF."""
        file_path_obj = Path(file_path)
        
        # Vérifier que le fichier existe
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas: {file_path}")
        
        if not file_path_obj.is_file():
            raise ValueError(f"Le chemin spécifié n'est pas un fichier: {file_path}")
        
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf n'est pas installé. Installez-le avec: pip install pypdf")
        
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                if len(pdf_reader.pages) == 0:
                    logger.warning(f"Le PDF {file_path} ne contient aucune page")
                    return ""
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                logger.warning(f"Aucun texte extrait du PDF {file_path}")
            
            return text
        except pypdf.errors.PdfReadError as e:
            logger.error(f"Erreur de lecture du PDF {file_path}: {e}")
            raise ValueError(f"Le fichier {file_path} n'est pas un PDF valide ou est corrompu.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du PDF {file_path}: {e}")
            raise
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """Charge un fichier Word (.docx)."""
        file_path_obj = Path(file_path)
        
        # Vérifier que le fichier existe
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas: {file_path}")
        
        if not file_path_obj.is_file():
            raise ValueError(f"Le chemin spécifié n'est pas un fichier: {file_path}")
        
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx n'est pas installé. Installez-le avec: pip install python-docx")
        
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            if not text.strip():
                logger.warning(f"Le document DOCX {file_path} ne contient aucun texte")
            
            return text
        except Exception as e:
            logger.error(f"Erreur lors du chargement du DOCX {file_path}: {e}")
            raise ValueError(f"Impossible de charger le fichier DOCX {file_path}. Vérifiez que c'est un fichier .docx valide.")
    
    @staticmethod
    def load_document(file_path: str) -> str:
        """
        Charge un document en fonction de son extension.
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format n'est pas supporté ou si le fichier est invalide
            PermissionError: Si l'accès au fichier est refusé
        """
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()
        
        # Vérifier le format avant de charger
        if extension not in DocumentLoader.SUPPORTED_FORMATS:
            supported = ", ".join(DocumentLoader.SUPPORTED_FORMATS)
            raise ValueError(
                f"Format de fichier non supporté: {extension}. "
                f"Formats supportés: {supported}"
            )
        
        # Charger selon le format
        if extension == '.txt' or extension == '.md':
            return DocumentLoader.load_text_file(str(file_path_obj))
        elif extension == '.pdf':
            return DocumentLoader.load_pdf(str(file_path_obj))
        elif extension == '.docx':
            return DocumentLoader.load_docx(str(file_path_obj))
        else:
            # Ne devrait jamais arriver grâce à la vérification précédente
            raise ValueError(f"Format non géré: {extension}")


class TextSplitter:
    """Classe pour découper le texte en chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialise le text splitter.
        
        Args:
            chunk_size: Taille maximale d'un chunk en caractères
            chunk_overlap: Nombre de caractères de chevauchement entre chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if RecursiveCharacterTextSplitter:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        else:
            self.splitter = None
    
    def split_text(self, text: str) -> List[str]:
        """Découpe le texte en chunks."""
        if self.splitter:
            return self.splitter.split_text(text)
        else:
            # Implémentation simple si langchain n'est pas disponible
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - self.chunk_overlap
            return chunks


class RAGManager:
    """
    Gestionnaire RAG générique pour charger, indexer et rechercher des documents.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        device: Optional[str] = None
    ):
        """
        Initialise le gestionnaire RAG.
        
        Args:
            collection_name: Nom de la collection ChromaDB
            persist_directory: Répertoire pour persister la base de données (None = mémoire)
            embedding_model: Nom du modèle d'embedding sentence-transformers
            chunk_size: Taille des chunks pour le découpage
            chunk_overlap: Chevauchement entre chunks
            device: Device pour les embeddings ('cuda', 'cpu', ou None pour auto)
        """
        if not chromadb:
            raise ImportError("chromadb n'est pas installé. Installez-le avec: pip install chromadb")
        
        if not SentenceTransformer:
            raise ImportError("sentence-transformers n'est pas installé. Installez-le avec: pip install sentence-transformers")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialiser le modèle d'embedding
        logger.info(f"Chargement du modèle d'embedding: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Initialiser ChromaDB
        self._init_chromadb()
        
        # Initialiser le text splitter
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        logger.info(f"RAGManager initialisé avec collection '{collection_name}'")
    
    def _init_chromadb(self):
        """Initialise la connexion à ChromaDB."""
        # Créer une fonction d'embedding compatible avec ChromaDB 0.4.16+
        embedding_function = SentenceTransformerEmbeddingFunction(self.embedding_model)
        
        # Configuration ChromaDB
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Créer ou récupérer la collection
        try:
            self.collection = client.get_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' récupérée")
        except Exception:
            self.collection = client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Collection '{self.collection_name}' créée")
    
    def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> int:
        """
        Ajoute un document à la base de connaissances.
        
        Args:
            file_path: Chemin vers le fichier à ajouter
            metadata: Métadonnées supplémentaires à associer au document
            document_id: ID unique pour le document (généré automatiquement si None)
        
        Returns:
            Nombre de chunks ajoutés
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format n'est pas supporté ou si le fichier est invalide
            PermissionError: Si l'accès au fichier est refusé
        """
        # Vérifier le format avant de charger
        if not DocumentLoader.is_supported_format(file_path):
            supported = DocumentLoader.get_supported_formats()
            raise ValueError(
                f"Format de fichier non supporté pour {file_path}. "
                f"Formats supportés: {', '.join(supported)}"
            )
        
        try:
            # Charger le document
            logger.info(f"Chargement du document: {file_path}")
            text = DocumentLoader.load_document(file_path)
            
            if not text.strip():
                logger.warning(f"Le document {file_path} est vide")
                return 0
            
            # Découper en chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Document découpé en {len(chunks)} chunks")
            
            if len(chunks) == 0:
                logger.warning(f"Aucun chunk généré pour le document {file_path}")
                return 0
            
            # Générer un ID de document si non fourni
            if document_id is None:
                document_id = hashlib.md5(file_path.encode()).hexdigest()
            
            # Préparer les métadonnées
            base_metadata = {
                "source": file_path,
                "file_name": Path(file_path).name,
            }
            if metadata:
                base_metadata.update(metadata)
            
            # Ajouter les chunks à ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk)
                
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                metadatas.append(chunk_metadata)
            
            # Ajouter à la collection
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout à ChromaDB: {e}")
                raise RuntimeError(f"Erreur lors de l'ajout des chunks à la base de données: {e}")
            
            logger.info(f"Document '{file_path}' ajouté avec {len(chunks)} chunks")
            return len(chunks)
            
        except (FileNotFoundError, ValueError, PermissionError):
            # Relancer les erreurs spécifiques telles quelles
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'ajout du document {file_path}: {e}")
            raise RuntimeError(f"Erreur lors de l'ajout du document {file_path}: {e}")
    
    def add_text(
        self,
        text: str,
        source: str = "manual_input",
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> int:
        """
        Ajoute du texte directement à la base de connaissances.
        
        Args:
            text: Texte à ajouter
            source: Source du texte
            metadata: Métadonnées supplémentaires
            document_id: ID unique pour le document
        
        Returns:
            Nombre de chunks ajoutés
        """
        if document_id is None:
            document_id = hashlib.md5(text.encode()).hexdigest()
        
        chunks = self.text_splitter.split_text(text)
        
        base_metadata = {"source": source}
        if metadata:
            base_metadata.update(metadata)
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            metadatas.append(chunk_metadata)
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Texte ajouté avec {len(chunks)} chunks")
        return len(chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche des documents pertinents pour une requête.
        
        Args:
            query: Requête de recherche
            n_results: Nombre de résultats à retourner
            filter_metadata: Filtres sur les métadonnées (ex: {"source": "file.pdf"})
        
        Returns:
            Liste de dictionnaires contenant 'document', 'metadata', 'distance'
        """
        try:
            where = filter_metadata if filter_metadata else None
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Formater les résultats
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            logger.info(f"Recherche '{query}' retourne {len(formatted_results)} résultats")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            raise
    
    def get_context_for_query(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Récupère le contexte formaté pour une requête (pour injection dans un prompt).
        
        Args:
            query: Requête de recherche
            n_results: Nombre de résultats à inclure
            filter_metadata: Filtres sur les métadonnées
        
        Returns:
            Contexte formaté en string
        """
        results = self.search(query, n_results=n_results, filter_metadata=filter_metadata)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            chunk_index = result['metadata'].get('chunk_index', '?')
            context_parts.append(
                f"[Document {i} - Source: {source}, Chunk {chunk_index}]\n{result['document']}"
            )
        
        return "\n\n".join(context_parts)
    
    def delete_collection(self):
        """Supprime la collection actuelle."""
        try:
            self.collection.delete()
            logger.info(f"Collection '{self.collection_name}' supprimée")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la collection: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Retourne des informations sur la collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "embedding_model": self.embedding_model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos: {e}")
            return {}

