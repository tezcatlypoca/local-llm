"""
Script de test rapide pour vérifier rapidement la vectorisation RAG.

Ce script peut être exécuté directement avec Python ou avec pytest.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGManager

# Texte de test simple
test_text = """
Python est un langage de programmation interprété, de haut niveau et à usage général.
Il est connu pour sa syntaxe simple et lisible. Python supporte plusieurs paradigmes
de programmation, notamment la programmation orientée objet, impérative, fonctionnelle
et procédurale.
"""


def test_quick_vectorization():
    """Test rapide de vectorisation."""
    print("Test rapide de vectorisation RAG...\n")
    
    # Créer une DB temporaire
    db_dir = tempfile.mkdtemp(prefix="rag_quick_test_")
    
    try:
        # Initialiser
        rag = RAGManager(
            collection_name="quick_test",
            persist_directory=db_dir,
            chunk_size=200,
            chunk_overlap=50
        )
        print("[OK] RAGManager initialise")
        
        # Ajouter du texte
        chunks = rag.add_text(test_text, source="test")
        print(f"[OK] {chunks} chunks crees")
        
        # Vérifier la collection
        info = rag.get_collection_info()
        print(f"[OK] Collection contient {info['total_chunks']} chunks")
        
        # Tester une recherche
        results = rag.search("langage de programmation", n_results=1)
        assert len(results) > 0, "Aucun résultat de recherche"
        print(f"[OK] Recherche fonctionne (distance: {results[0]['distance']:.4f})")
        
        # Vérifier les embeddings
        embedding = rag.embedding_model.encode("test", convert_to_numpy=True)
        assert len(embedding) == 384, f"Dimension attendue: 384, obtenue: {len(embedding)}"
        print(f"[OK] Embeddings generes (dimension: {len(embedding)})")
        
        print("\n[SUCCES] Test rapide reussi !")
        return True
        
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Exécution directe avec Python
    success = test_quick_vectorization()
    sys.exit(0 if success else 1)
