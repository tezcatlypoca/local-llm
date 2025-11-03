#!/usr/bin/env python3
"""
Script pour télécharger un petit modèle LLM pour tester l'API.
"""
import os
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

# Ajouter le dossier src au path pour les imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
except ImportError:
    print("❌ Erreur: transformers n'est pas installé.")
    print("   Installez les dépendances avec: pip install -r requirements.txt")
    sys.exit(1)

# Essayer d'importer BitsAndBytes pour la quantisation (optionnel)
# Note: BitsAndBytes n'est PAS compatible avec ROCm/AMD, seulement CUDA/NVIDIA
BITSANDBYTES_AVAILABLE = False
BITSANDBYTES_CONFIG = None

# Détecter si on est sur ROCm (AMD) ou CUDA (NVIDIA)
IS_ROCM = False
if torch is not None:
    try:
        # Sur ROCm, torch.version.hip existe et n'est pas None
        IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None
    except:
        pass

# Si on n'est pas sur ROCm, on peut essayer d'importer bitsandbytes
if not IS_ROCM:
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        BITSANDBYTES_AVAILABLE = True
        BITSANDBYTES_CONFIG = BitsAndBytesConfig
    except (ImportError, ModuleNotFoundError, Exception) as e:
        # L'import peut échouer pour diverses raisons
        BITSANDBYTES_AVAILABLE = False
        BITSANDBYTES_CONFIG = None


def download_model(model_name: str = "gpt2", use_quantization: bool = False):
    """
    Télécharge un modèle Hugging Face.
    
    Args:
        model_name: Nom du modèle à télécharger (défaut: "gpt2")
        use_quantization: Utiliser la quantisation 4-bit si disponible (pour les grands modèles)
    """
    print(f"📥 Téléchargement du modèle '{model_name}'...")
    print("   (Ceci peut prendre quelques minutes selon votre connexion)\n")
    
    # Détecter le type de modèle basé sur le nom
    is_classification_model = any(x in model_name.lower() for x in ["finbert", "bert", "classifier"])
    
    try:
        # Téléchargement du tokenizer
        print(f"1/2 Téléchargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("   ✅ Tokenizer téléchargé\n")
        
        # Téléchargement du modèle
        print(f"2/2 Téléchargement du modèle...")
        
        # Configuration de la quantisation si demandée et disponible
        load_kwargs = {}
        if use_quantization and BITSANDBYTES_AVAILABLE and not is_classification_model:
            try:
                print("   📊 Utilisation de la quantisation 4-bit (BitsAndBytes)...")
                # Utiliser torch.float16 si disponible, sinon "float16" (string)
                compute_dtype = torch.float16 if torch is not None else "float16"
                quantization_config = BITSANDBYTES_CONFIG(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
            except Exception as e:
                print(f"   ⚠️  Erreur lors de la configuration de BitsAndBytes: {e}")
                if IS_ROCM:
                    print("   ℹ️  BitsAndBytes n'est pas compatible avec ROCm/AMD")
                print("   📥 Téléchargement en full precision...")
                use_quantization = False  # Désactiver pour éviter d'autres erreurs
        elif use_quantization and not BITSANDBYTES_AVAILABLE:
            if IS_ROCM:
                print("   ⚠️  BitsAndBytes n'est pas compatible avec ROCm/AMD")
                print("   ℹ️  Pour ROCm, considérez d'utiliser des modèles pré-quantifiés (GPTQ/AWQ)")
            else:
                print("   ⚠️  BitsAndBytes non disponible. Installation: pip install bitsandbytes")
            print("   📥 Téléchargement en full precision...")
        
        # Sélectionner la classe de modèle appropriée
        if is_classification_model:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        print("   ✅ Modèle téléchargé\n")
        
        # Les fichiers sont automatiquement mis en cache dans ~/.cache/huggingface/hub
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"✅ Modèle '{model_name}' téléchargé avec succès !")
        print(f"   Cache: {cache_dir}")
        print(f"   Type: {'Classification' if is_classification_model else 'Génératif'}")
        if use_quantization and BITSANDBYTES_AVAILABLE:
            print(f"   Quantification: 4-bit activée")
        
        print(f"\n💡 Vous pouvez maintenant tester l'API avec:")
        print(f"   GET http://localhost:5000/models")
        print(f"\n   Le modèle apparaîtra dans la liste des modèles disponibles.")
        
        if is_classification_model:
            print(f"\n⚠️  NOTE: FinBERT est un modèle de classification (BERT), pas un modèle génératif.")
            print(f"   Il nécessitera des routes API spécifiques pour l'analyse de sentiment/classification.")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Erreur lors du téléchargement: {error_msg}")
        
        # Détecter les erreurs courantes et donner des suggestions
        if "is not a valid model identifier" in error_msg or "not a local folder" in error_msg:
            print("\n⚠️  Le nom du modèle semble incorrect.")
            print("   Vérifiez le nom exact sur: https://huggingface.co/models")
            print("   Exemples valides:")
            print("   - Qwen/Qwen2.5-1.5B-Instruct")
            print("   - Qwen/Qwen2.5-3B-Instruct")
            print("   - Qwen/Qwen2-1.5B-Instruct")
        
        print("\n💡 Vérifiez également:")
        print("   - Votre connexion internet")
        print("   - Que transformers est correctement installé")
        print("   - Que vous avez suffisamment d'espace disque")
        print("   - Si c'est un modèle privé, connectez-vous avec: huggingface-cli login")
        
        return False


def main():
    """Point d'entrée principal."""
    print("=" * 60)
    print("Téléchargement d'un modèle de test pour l'API LLM")
    print("=" * 60)
    print()
    
    # Informer l'utilisateur si ROCm est détecté
    if IS_ROCM:
        print("ℹ️  ROCm détecté: bitsandbytes n'est pas compatible avec AMD/ROCm")
        print("   La quantisation 4-bit via bitsandbytes sera désactivée.")
        print("   Pour ROCm, considérez d'utiliser des modèles pré-quantifiés (GPTQ/AWQ).\n")
    
    # Liste de modèles recommandés (du plus petit au plus grand)
    models = {
        "1": ("gpt2", "GPT2 - ~500 MB - Très rapide, bon pour les tests", False),
        "2": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama Chat - ~2.2 GB - Modèle conversationnel", False),
        "3": ("Qwen/Qwen2-1.5B-Instruct", "Qwen2 1.5B - ~3 GB - Modèle instruct/chat multilingue", False),
        "4": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B - ~3 GB - ⭐ RECOMMANDÉ - Meilleur compromis qualité/taille (8GB)", False),
        "5": ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B - ~6 GB - Plus performant (limite 8GB)", False),
        "6": ("microsoft/phi-2", "Phi-2 - ~5.4 GB - Modèle Microsoft performant (attention: limite 8GB)", False),
        "7": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B - ~14 GB (4-5 GB quantifié) - ⭐ PROCHAIN GPT-4 - Meilleure qualité", True),
        "8": ("ProsusAI/finbert", "FinBERT - ~0.44 GB - Modèle financier (classification, pas génératif)", False),
    }
    
    print("Modèles disponibles pour téléchargement:")
    print()
    for key, (model_name, description, needs_quant) in models.items():
        quant_note = " (nécessite quantisation 4-bit)" if needs_quant else ""
        print(f"  {key}. {description}{quant_note}")
    print()
    
    choice = input("Choisissez un modèle (1-8) ou entrez un nom de modèle Hugging Face: ").strip()
    
    use_quantization = False
    if choice in models:
        model_name, description, needs_quant = models[choice]
        if needs_quant:
            print(f"\n⚠️  Le modèle '{model_name}' fait ~14 GB en full precision.")
            print(f"   Pour tenir dans 8 GB de VRAM, la quantisation 4-bit est recommandée.")
            if BITSANDBYTES_AVAILABLE:
                quant_choice = input("   Utiliser la quantisation 4-bit ? (O/n): ").strip().lower()
                use_quantization = quant_choice != 'n'
            else:
                if IS_ROCM:
                    print(f"   ⚠️  BitsAndBytes n'est pas compatible avec ROCm/AMD.")
                    print(f"   ℹ️  Pour ROCm, utilisez des modèles pré-quantifiés (GPTQ/AWQ) ou téléchargez en full precision.")
                    print(f"   📥 Téléchargement en full precision (nécessitera plus de 8 GB)...")
                else:
                    print(f"   ⚠️  BitsAndBytes n'est pas installé. Installation: pip install bitsandbytes")
                    print(f"   📥 Téléchargement en full precision (nécessitera plus de 8 GB)...")
                use_quantization = False
    elif choice:
        model_name = choice
        # Détecter si c'est un grand modèle qui pourrait bénéficier de la quantisation
        if "7b" in choice.lower() or "8b" in choice.lower() or "13b" in choice.lower():
            print(f"\n⚠️  Grand modèle détecté. Quantification recommandée pour 8GB VRAM.")
            if BITSANDBYTES_AVAILABLE:
                quant_choice = input("   Utiliser la quantisation 4-bit ? (O/n): ").strip().lower()
                use_quantization = quant_choice != 'n'
            elif IS_ROCM:
                print(f"   ℹ️  BitsAndBytes n'est pas compatible avec ROCm/AMD.")
                print(f"   Téléchargement en full precision (nécessitera plus de 8 GB)...")
    else:
        # Par défaut : Qwen2.5-1.5B-Instruct (recommandé pour 8GB)
        print("Utilisation du modèle par défaut recommandé: Qwen/Qwen2.5-1.5B-Instruct")
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print()
    success = download_model(model_name, use_quantization=use_quantization)
    
    if success:
        print("\n✅ Téléchargement terminé !")
        sys.exit(0)
    else:
        print("\n❌ Échec du téléchargement.")
        sys.exit(1)


if __name__ == "__main__":
    main()

