#!/usr/bin/env python3
"""
Gestionnaire de modèles LLM au format GGUF pour utilisation avec llama.cpp.
Ce module permet de charger et gérer des modèles quantifiés GGUF.
"""

import os
import gc
import logging
import secrets
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMGGUFManager:
    """Gestionnaire pour charger et utiliser des modèles LLM au format GGUF."""
    
    def __init__(self):
        """Initialise le gestionnaire de modèles GGUF."""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python n'est pas installé. Installez-le avec: pip install llama-cpp-python")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers n'est pas installé pour charger les tokenizers.")
        
        # Dictionnaires pour stocker les modèles et tokenizers par GPU
        self.models: Dict[int, Optional[Any]] = {0: None, 1: None}
        self.tokenizers: Dict[int, Optional[Any]] = {0: None, 1: None}
        self.model_paths: Dict[int, Optional[str]] = {0: None, 1: None}
        self.model_names: Dict[int, Optional[str]] = {0: None, 1: None}
        self.tokenizer_names: Dict[int, Optional[str]] = {0: None, 1: None}
        # Tokens d'accès pour sécuriser le déchargement
        self.access_tokens: Dict[int, Optional[str]] = {0: None, 1: None}
    
    def _find_gguf_file(self, model_path: str) -> Optional[str]:
        """
        Trouve le fichier .gguf dans le chemin donné.
        
        Args:
            model_path: Chemin vers le modèle (fichier .gguf ou répertoire)
        
        Returns:
            Chemin complet vers le fichier .gguf ou None si introuvable
        """
        path = Path(model_path)
        
        # Si c'est déjà un fichier .gguf
        if path.is_file() and path.suffix == '.gguf':
            return str(path.absolute())
        
        # Si c'est un répertoire, chercher un fichier .gguf dedans
        if path.is_dir():
            gguf_files = list(path.glob('*.gguf'))
            if gguf_files:
                # Préférer les fichiers Q4_K_M, puis Q4_0, puis le premier trouvé
                preferred = [f for f in gguf_files if 'q4_k_m' in f.name.lower()]
                if not preferred:
                    preferred = [f for f in gguf_files if 'q4_0' in f.name.lower()]
                if preferred:
                    return str(preferred[0].absolute())
                return str(gguf_files[0].absolute())
        
        # Chercher dans le cache Hugging Face
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache_dir):
            # Chercher récursivement
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.gguf') and model_path.replace('/', '--') in root:
                        return os.path.join(root, file)
        
        return None
    
    def _infer_tokenizer_name(self, model_path: str) -> Optional[str]:
        """
        Infère le nom du tokenizer à partir du chemin du modèle GGUF.
        
        Args:
            model_path: Chemin vers le modèle GGUF
        
        Returns:
            Nom du modèle Hugging Face pour le tokenizer ou None
        """
        path_lower = model_path.lower()
        
        # Mapping des modèles GGUF vers leurs tokenizers
        if 'qwen2.5-7b-instruct' in path_lower or 'qwen2.5_7b_instruct' in path_lower:
            return "Qwen/Qwen2.5-7B-Instruct"
        elif 'qwen2-7b-instruct' in path_lower:
            return "Qwen/Qwen2-7B-Instruct"
        elif 'mistral-7b-instruct' in path_lower or 'mistral_7b_instruct' in path_lower:
            return "mistralai/Mistral-7B-Instruct-v0.2"
        elif 'mistral-7b-v0.1' in path_lower:
            return "mistralai/Mistral-7B-v0.1"
        
        # Essayer d'extraire depuis le chemin
        parts = model_path.replace('\\', '/').split('/')
        for part in parts:
            if 'qwen' in part.lower():
                if '2.5' in part.lower():
                    return "Qwen/Qwen2.5-7B-Instruct"
                return "Qwen/Qwen2-7B-Instruct"
            elif 'mistral' in part.lower():
                return "mistralai/Mistral-7B-Instruct-v0.2"
        
        return None
    
    def load_model(self, model_path: str, gpu_id: int = 0, tokenizer_name: Optional[str] = None, **model_kwargs) -> tuple[bool, Optional[str]]:
        """
        Charge un modèle GGUF sur le GPU spécifié.
        
        Args:
            model_path: Chemin vers le fichier .gguf ou répertoire contenant un .gguf
            gpu_id: ID du GPU (0 ou 1) sur lequel charger le modèle
            tokenizer_name: Nom du tokenizer Hugging Face (optionnel, sera inféré si non fourni)
            **model_kwargs: Arguments additionnels pour llama.cpp
                - n_ctx: Taille du contexte (défaut: 2048)
                - n_gpu_layers: Nombre de couches sur GPU, -1 = toutes (défaut: -1)
                - n_threads: Nombre de threads CPU (défaut: auto)
        
        Returns:
            (success: bool, access_token: Optional[str]) - True et le token si succès, False et None sinon
        """
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python n'est pas installé. Installez-le avec: pip install llama-cpp-python")
            return False, None
        
        if gpu_id not in [0, 1]:
            logger.error(f"GPU ID invalide: {gpu_id}. Doit être 0 ou 1.")
            return False, None
        
        if self.models[gpu_id] is not None:
            logger.warning(f"Un modèle est déjà chargé sur GPU {gpu_id}. Déchargez-le d'abord.")
            return False, None
        
        logger.info(f"Chargement du modèle GGUF '{model_path}' sur GPU {gpu_id}...")
        
        try:
            # Trouver le fichier .gguf
            gguf_file = self._find_gguf_file(model_path)
            if gguf_file is None:
                logger.error(f"Fichier .gguf introuvable pour '{model_path}'")
                return False, None
            
            logger.info(f"Fichier GGUF trouvé: {gguf_file}")
            
            # Inférer le nom du tokenizer si non fourni
            if tokenizer_name is None:
                tokenizer_name = self._infer_tokenizer_name(model_path)
                if tokenizer_name:
                    logger.info(f"Tokenizer inféré: {tokenizer_name}")
                else:
                    logger.warning(f"Impossible d'inférer le tokenizer. Utilisation sans tokenizer externe.")
            
            # Charger le tokenizer si disponible
            tokenizer = None
            if tokenizer_name and TRANSFORMERS_AVAILABLE:
                try:
                    logger.info(f"Chargement du tokenizer '{tokenizer_name}'...")
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Tokenizer '{tokenizer_name}' chargé avec succès")
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement du tokenizer '{tokenizer_name}': {e}")
                    logger.info("Continuation sans tokenizer externe (llama.cpp utilisera son tokenizer intégré)")
            
            # Paramètres par défaut pour llama.cpp
            n_ctx = model_kwargs.get('n_ctx', 2048)
            n_gpu_layers = model_kwargs.get('n_gpu_layers', -1)  # -1 = toutes les couches sur GPU
            n_threads = model_kwargs.get('n_threads', None)
            verbose = model_kwargs.get('verbose', False)
            
            logger.info(f"Chargement du modèle GGUF avec n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}...")
            
            # Charger le modèle avec llama-cpp-python
            llama_kwargs = {
                'model_path': gguf_file,
                'n_ctx': n_ctx,
                'n_gpu_layers': n_gpu_layers,
                'verbose': verbose
            }
            if n_threads is not None:
                llama_kwargs['n_threads'] = n_threads
            
            model = Llama(**llama_kwargs)
            
            logger.info(f"Modèle GGUF chargé avec succès sur GPU {gpu_id}")
            
            # Stockage
            self.models[gpu_id] = model
            self.tokenizers[gpu_id] = tokenizer
            self.model_paths[gpu_id] = gguf_file
            self.model_names[gpu_id] = model_path
            self.tokenizer_names[gpu_id] = tokenizer_name
            
            # Génération d'un token d'accès unique
            access_token = secrets.token_urlsafe(32)
            self.access_tokens[gpu_id] = access_token
            
            logger.info(f"Modèle '{model_path}' chargé avec succès sur GPU {gpu_id}")
            logger.debug(f"Token d'accès généré pour GPU {gpu_id}: {access_token[:16]}...")
            
            return True, access_token
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Erreur lors du chargement du modèle GGUF '{model_path}': {error_msg}", exc_info=True)
            
            # Nettoyage en cas d'erreur
            self.models[gpu_id] = None
            self.tokenizers[gpu_id] = None
            self.model_paths[gpu_id] = None
            self.model_names[gpu_id] = None
            self.tokenizer_names[gpu_id] = None
            self.access_tokens[gpu_id] = None
            return False, None
    
    def unload_model(self, gpu_id: int, access_token: Optional[str] = None) -> tuple[bool, str]:
        """Décharge un modèle du GPU spécifié."""
        if gpu_id not in [0, 1]:
            return False, f"GPU ID invalide: {gpu_id}. Doit être 0 ou 1."
        
        if self.models[gpu_id] is None:
            return False, f"Aucun modèle chargé sur GPU {gpu_id}."
        
        # Vérification du token d'accès
        if access_token is None:
            return False, "Token d'accès requis pour décharger le modèle."
        
        if self.access_tokens[gpu_id] is None:
            return False, f"Aucun token d'accès associé au GPU {gpu_id}."
        
        if self.access_tokens[gpu_id] != access_token:
            logger.warning(f"Tentative de déchargement avec un token invalide sur GPU {gpu_id}")
            return False, "Token d'accès invalide."
        
        model_name = self.model_names[gpu_id]
        logger.info(f"Déchargement du modèle '{model_name}' du GPU {gpu_id}...")
        
        try:
            # Suppression du modèle
            del self.models[gpu_id]
            self.models[gpu_id] = None
            self.tokenizers[gpu_id] = None
            self.model_paths[gpu_id] = None
            self.model_names[gpu_id] = None
            self.tokenizer_names[gpu_id] = None
            self.access_tokens[gpu_id] = None
            
            # Nettoyage de la mémoire
            gc.collect()
            
            logger.info(f"Modèle '{model_name}' déchargé avec succès.")
            return True, f"Modèle '{model_name}' déchargé avec succès du GPU {gpu_id}."
            
        except Exception as e:
            logger.error(f"Erreur lors du déchargement du modèle: {str(e)}", exc_info=True)
            return False, f"Erreur lors du déchargement: {str(e)}"
    
    def generate(
        self,
        prompt: str,
        gpu_id: int = 0,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        **generation_kwargs
    ) -> Optional[str]:
        """
        Génère une réponse à partir d'un prompt en utilisant le modèle GGUF.
        
        Args:
            prompt: Texte d'entrée
            gpu_id: ID du GPU (0 ou 1) à utiliser
            max_new_tokens: Nombre maximum de nouveaux tokens à générer
            temperature: Température pour la génération
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            repetition_penalty: Pénalité de répétition
            **generation_kwargs: Arguments additionnels pour llama.cpp
        
        Returns:
            Texte généré ou None en cas d'erreur
        """
        if gpu_id not in [0, 1]:
            logger.error(f"GPU ID invalide: {gpu_id}. Doit être 0 ou 1.")
            return None
        
        if self.models[gpu_id] is None:
            logger.error(f"Aucun modèle chargé sur GPU {gpu_id}.")
            return None
        
        model = self.models[gpu_id]
        
        try:
            # Génération avec llama.cpp
            output = model(
                prompt.strip(),
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stop=None,  # Pas de stop tokens par défaut
                **generation_kwargs
            )
            
            # Extraire le texte généré
            if 'choices' in output and len(output['choices']) > 0:
                generated_text = output['choices'][0]['text']
                logger.info(f"Texte généré (longueur: {len(generated_text)}): {generated_text[:200]}...")
                return generated_text.strip()
            else:
                logger.error(f"Format de réponse inattendu de llama.cpp: {output}")
                return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}", exc_info=True)
            return None
    
    def get_model_status(self, gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """Retourne le statut des modèles chargés."""
        if gpu_id is not None:
            if gpu_id not in [0, 1]:
                return {"error": f"GPU ID invalide: {gpu_id}"}
            
            return {
                "gpu_id": gpu_id,
                "model_loaded": self.models[gpu_id] is not None,
                "model_name": self.model_names[gpu_id],
                "model_path": self.model_paths[gpu_id],
                "tokenizer_name": self.tokenizer_names[gpu_id],
                "model_type": "gguf"
            }
        
        # Statut de tous les GPUs
        status = {
            "total_gpus": 2,  # Fixe à 2 pour compatibilité
            "gpus": {},
            "model_type": "gguf"
        }
        
        for gpu_id in [0, 1]:
            status["gpus"][gpu_id] = {
                "model_loaded": self.models[gpu_id] is not None,
                "model_name": self.model_names[gpu_id],
                "model_path": self.model_paths[gpu_id],
                "tokenizer_name": self.tokenizer_names[gpu_id],
                "has_access_token": self.access_tokens[gpu_id] is not None
            }
        
        return status

