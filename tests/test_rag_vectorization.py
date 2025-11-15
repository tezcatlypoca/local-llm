"""
Tests pytest pour vérifier que le système RAG vectorise correctement les documents.
"""

import os
import tempfile
import pytest
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGManager

# Document de test
TEST_DOCUMENT = """
Introduction à l'intelligence artificielle

L'intelligence artificielle (IA) est une branche de l'informatique qui vise à créer 
des systèmes capables de simuler l'intelligence humaine. Les domaines de l'IA incluent 
l'apprentissage automatique, le traitement du langage naturel, la vision par ordinateur, 
et la robotique.

Apprentissage automatique

L'apprentissage automatique (machine learning) est une sous-catégorie de l'IA qui permet 
aux systèmes d'apprendre et de s'améliorer à partir de données sans être explicitement 
programmés. Il existe trois types principaux d'apprentissage :
- L'apprentissage supervisé : le modèle apprend à partir de données étiquetées
- L'apprentissage non supervisé : le modèle trouve des patterns dans des données non étiquetées
- L'apprentissage par renforcement : le modèle apprend par essais et erreurs avec des récompenses

Traitement du langage naturel

Le traitement du langage naturel (NLP) permet aux machines de comprendre, interpréter 
et générer du langage humain. Les applications incluent les chatbots, la traduction 
automatique, l'analyse de sentiment, et les assistants vocaux.
"""


@pytest.fixture
def temp_db_dir():
    """Crée un répertoire temporaire pour la base de données."""
    db_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield db_dir
    # Nettoyage après le test
    import shutil
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir, ignore_errors=True)


@pytest.fixture
def rag_manager(temp_db_dir):
    """Crée une instance de RAGManager pour les tests."""
    return RAGManager(
        collection_name="test_collection",
        persist_directory=temp_db_dir,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100,
        device=None
    )


@pytest.fixture
def test_file(temp_db_dir):
    """Crée un fichier de test."""
    test_file_path = os.path.join(temp_db_dir, "test_document.txt")
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(TEST_DOCUMENT)
    return test_file_path


class TestRAGVectorization:
    """Tests de vectorisation RAG."""
    
    def test_rag_manager_initialization(self, rag_manager):
        """Test que le RAGManager s'initialise correctement."""
        assert rag_manager is not None
        assert rag_manager.embedding_model_name == "all-MiniLM-L6-v2"
        assert rag_manager.chunk_size == 500
        assert rag_manager.chunk_overlap == 100
    
    def test_add_document_creates_chunks(self, rag_manager, test_file):
        """Test que l'ajout d'un document crée des chunks."""
        chunks_count = rag_manager.add_document(test_file)
        assert chunks_count > 0
        
        info = rag_manager.get_collection_info()
        assert info['total_chunks'] == chunks_count
    
    def test_add_text_creates_chunks(self, rag_manager):
        """Test que l'ajout de texte crée des chunks."""
        test_text = "Python est un langage de programmation. Il est simple et puissant."
        chunks_count = rag_manager.add_text(test_text, source="test")
        assert chunks_count > 0
        
        info = rag_manager.get_collection_info()
        assert info['total_chunks'] == chunks_count
    
    def test_embeddings_are_generated(self, rag_manager, test_file):
        """Test que les embeddings sont générés et stockés."""
        rag_manager.add_document(test_file)
        
        # Vérifier qu'on peut récupérer les embeddings via une recherche
        results = rag_manager.collection.query(
            query_texts=["intelligence artificielle"],
            n_results=1,
            include=['embeddings', 'documents']
        )
        
        assert results['embeddings'] is not None
        assert len(results['embeddings'][0]) > 0
        
        embedding = results['embeddings'][0][0]
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_search_returns_results(self, rag_manager, test_file):
        """Test que la recherche vectorielle retourne des résultats."""
        rag_manager.add_document(test_file)
        
        results = rag_manager.search("apprentissage automatique", n_results=3)
        
        assert len(results) > 0
        assert all('document' in r for r in results)
        assert all('metadata' in r for r in results)
        assert all('distance' in r for r in results)
    
    def test_search_results_have_correct_structure(self, rag_manager, test_file):
        """Test que les résultats de recherche ont la structure correcte."""
        rag_manager.add_document(test_file)
        
        results = rag_manager.search("machine learning", n_results=1)
        
        assert len(results) == 1
        result = results[0]
        
        assert 'document' in result
        assert isinstance(result['document'], str)
        assert len(result['document']) > 0
        
        assert 'metadata' in result
        assert isinstance(result['metadata'], dict)
        assert 'source' in result['metadata']
        
        assert 'distance' in result
        assert isinstance(result['distance'], (int, float))
        assert result['distance'] >= 0
    
    def test_embedding_dimension(self, rag_manager):
        """Test que les embeddings ont la dimension attendue."""
        # all-MiniLM-L6-v2 produit des embeddings de 384 dimensions
        test_text = "Test d'embedding"
        embedding = rag_manager.embedding_model.encode(test_text, convert_to_numpy=True)
        
        assert embedding.shape == (384,)
    
    def test_collection_info(self, rag_manager, test_file):
        """Test que get_collection_info retourne les bonnes informations."""
        chunks_count = rag_manager.add_document(test_file)
        
        info = rag_manager.get_collection_info()
        
        assert info['collection_name'] == "test_collection"
        assert info['total_chunks'] == chunks_count
        assert info['embedding_model'] == "all-MiniLM-L6-v2"
        assert info['chunk_size'] == 500
        assert info['chunk_overlap'] == 100
    
    def test_search_coherence(self, rag_manager, test_file):
        """Test que la recherche peut retrouver les chunks originaux."""
        rag_manager.add_document(test_file)
        
        # Récupérer un chunk original
        all_chunks = rag_manager.collection.get()
        assert len(all_chunks['documents']) > 0
        
        first_chunk_text = all_chunks['documents'][0]
        # Rechercher avec le début du chunk
        search_results = rag_manager.search(first_chunk_text[:50], n_results=1)
        
        assert len(search_results) > 0
        # La distance devrait être très faible pour le chunk original
        assert search_results[0]['distance'] < 0.5
    
    def test_multiple_documents(self, rag_manager, temp_db_dir):
        """Test l'ajout de plusieurs documents."""
        # Ajouter plusieurs textes
        texts = [
            "Premier document sur Python",
            "Deuxième document sur JavaScript",
            "Troisième document sur Java"
        ]
        
        total_chunks = 0
        for text in texts:
            chunks = rag_manager.add_text(text, source=f"doc_{texts.index(text)}")
            total_chunks += chunks
        
        info = rag_manager.get_collection_info()
        assert info['total_chunks'] == total_chunks
        
        # Vérifier qu'on peut rechercher dans tous les documents
        results = rag_manager.search("programmation", n_results=5)
        assert len(results) > 0
