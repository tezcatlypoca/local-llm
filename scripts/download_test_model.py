#!/usr/bin/env python3
"""
Script pour télécharger un petit modèle LLM pour tester l'API.
"""
import os
import sys
from pathlib import Path

# Ajouter le dossier src au path pour les imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("❌ Erreur: transformers n'est pas installé.")
    print("   Installez les dépendances avec: pip install -r requirements.txt")
    sys.exit(1)


def download_model(model_name: str = "gpt2"):
    """
    Télécharge un modèle Hugging Face.
    
    Args:
        model_name: Nom du modèle à télécharger (défaut: "gpt2")
    """
    print(f"📥 Téléchargement du modèle '{model_name}'...")
    print("   (Ceci peut prendre quelques minutes selon votre connexion)\n")
    
    try:
        # Téléchargement du tokenizer
        print(f"1/2 Téléchargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("   ✅ Tokenizer téléchargé\n")
        
        # Téléchargement du modèle
        print(f"2/2 Téléchargement du modèle...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("   ✅ Modèle téléchargé\n")
        
        # Les fichiers sont automatiquement mis en cache dans ~/.cache/huggingface/hub
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"✅ Modèle '{model_name}' téléchargé avec succès !")
        print(f"   Cache: {cache_dir}")
        print(f"\n💡 Vous pouvez maintenant tester l'API avec:")
        print(f"   GET http://localhost:5000/models")
        print(f"\n   Le modèle apparaîtra dans la liste des modèles disponibles.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement: {e}")
        print("\n💡 Vérifiez:")
        print("   - Votre connexion internet")
        print("   - Que transformers est correctement installé")
        print("   - Que vous avez suffisamment d'espace disque")
        return False


def main():
    """Point d'entrée principal."""
    print("=" * 60)
    print("Téléchargement d'un modèle de test pour l'API LLM")
    print("=" * 60)
    print()
    
    # Liste de modèles recommandés (du plus petit au plus grand)
    models = {
        "1": ("gpt2", "GPT2 - ~500 MB - Très rapide, bon pour les tests"),
        "2": ("tinyllama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama - ~2 GB - Modèle conversationnel"),
        "3": ("microsoft/phi-2", "Phi-2 - ~5 GB - Modèle Microsoft performant"),
    }
    
    print("Modèles disponibles pour téléchargement:")
    print()
    for key, (model_name, description) in models.items():
        print(f"  {key}. {description}")
    print()
    
    choice = input("Choisissez un modèle (1-3) ou entrez un nom de modèle Hugging Face: ").strip()
    
    if choice in models:
        model_name = models[choice][0]
    elif choice:
        model_name = choice
    else:
        print("Utilisation du modèle par défaut: gpt2")
        model_name = "gpt2"
    
    print()
    success = download_model(model_name)
    
    if success:
        print("\n✅ Téléchargement terminé !")
        sys.exit(0)
    else:
        print("\n❌ Échec du téléchargement.")
        sys.exit(1)


if __name__ == "__main__":
    main()

