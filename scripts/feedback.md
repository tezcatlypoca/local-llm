(rocm_env) rhododendron@la-grosse-bertha:~/Documents/local-llm$ /home/rhododendron/venvs/rocm_env/bin/python /home/rhododendron/Documents/local-llm/scripts/download_test_model.py 
============================================================
Téléchargement d'un modèle de test pour l'API LLM
============================================================

Quel type de modèle souhaitez-vous télécharger ?

  1. Modèles standard (transformers) - Compatibles avec l'API actuelle
  2. Modèles quantifiés GGUF - Nécessitent llama.cpp (recommandé pour Q4)

Choisissez une option (1 ou 2) [défaut: 1]: 2

============================================================
📦 Téléchargement de modèles quantifiés GGUF
============================================================

ℹ️  Les modèles GGUF sont pré-quantifiés et optimisés.
   Avantages: Téléchargement 3x plus rapide, compatible CPU/GPU, meilleure performance.
   Note: Ces modèles nécessitent llama.cpp pour être utilisés (pas encore intégré dans l'API).

ℹ️  ROCm détecté: Les modèles GGUF sont parfaitement compatibles avec AMD/ROCm !

Modèles GGUF quantifiés disponibles:

  1. Qwen2.5 7B Instruct Q4_K_M - ~4.5 GB - ⭐ RECOMMANDÉ - Excellente qualité, optimisé pour 8GB VRAM
  2. Mistral 7B Instruct v0.2 Q4_K_M - ~4.5 GB - Modèle performant, optimisé pour 8GB VRAM
  3. Qwen2.5 7B Instruct Q4_0 - ~4.0 GB - Version plus petite (qualité légèrement inférieure)
  4. Mistral 7B Instruct v0.2 Q4_0 - ~4.0 GB - Version plus petite

Choisissez un modèle (1-4) ou entrez un nom personnalisé (format: repo_id/filename.gguf): 1

📥 Téléchargement du modèle GGUF quantifié...
   Dépôt: Qwen/Qwen2.5-7B-Instruct-GGUF
   Fichier: qwen2.5-7b-instruct-q4_k_m.gguf
   (Ceci peut prendre quelques minutes selon votre connexion)

1/1 Téléchargement du fichier GGUF...
/home/rhododendron/venvs/rocm_env/lib/python3.12/site-packages/huggingface_hub/file_download.py:979: UserWarning: `local_dir_use_symlinks` parameter is deprecated and will be ignored. The process to download files to a local folder has been updated and do not rely on symlinks anymore. You only need to pass a destination folder as`local_dir`.
For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/download#download-files-to-local-folder.
  warnings.warn(

❌ Erreur lors du téléchargement: 404 Client Error. (Request ID: Root=1-690a6998-09ac98001c9abe2d26883a0b;7aa9d429-21b6-4432-9c73-4fb38633d215)

Entry Not Found for url: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf.

⚠️  Le fichier ou le dépôt semble introuvable.
   Vérifiez que:
   - Le dépôt 'Qwen/Qwen2.5-7B-Instruct-GGUF' existe sur Hugging Face
   - Le fichier 'qwen2.5-7b-instruct-q4_k_m.gguf' existe dans ce dépôt
   - Visitez: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF

💡 Vérifiez également:
   - Votre connexion internet
   - Que huggingface_hub est correctement installé
   - Que vous avez suffisamment d'espace disque
   - Si c'est un modèle privé, connectez-vous avec: huggingface-cli login

❌ Échec du téléchargement.