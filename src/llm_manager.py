#!/usr/bin/env python3
"""
Gestionnaire de modèles LLM pour utilisation avec GPUs AMD via PyTorch/ROCm.

Ce module permet de charger et gérer jusqu'à 2 modèles LLM simultanément,
un sur chaque GPU AMD disponible.
"""

import os
import gc
import logging
import secrets
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMManager:
    """Gestionnaire pour charger et utiliser des modèles LLM sur GPUs AMD."""
    
    def __init__(self, miopen_cache_size: int = 4096):
        """
        Initialise le gestionnaire de modèles LLM.
        
        Args:
            miopen_cache_size: Taille maximale du cache MIOpen en Mo (défaut: 4096)
        """
        # Configuration du cache MIOpen (identique à test_run_model.py)
        self._setup_miopen_cache(miopen_cache_size)
        
        # Vérification de la disponibilité des GPUs
        if not torch.cuda.is_available():
            logger.warning("Aucun GPU CUDA détecté. Le CPU sera utilisé.")
            self.num_gpus = 0
        else:
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Nombre de GPUs détectés: {self.num_gpus}")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} Go)")
        
        # Dictionnaires pour stocker les modèles et tokenizers par GPU
        self.models: Dict[int, Optional[Any]] = {0: None, 1: None}
        self.tokenizers: Dict[int, Optional[Any]] = {0: None, 1: None}
        self.model_names: Dict[int, Optional[str]] = {0: None, 1: None}
        # Tokens d'accès pour sécuriser le déchargement (un token par GPU)
        self.access_tokens: Dict[int, Optional[str]] = {0: None, 1: None}
        self.devices: Dict[int, torch.device] = {
            0: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            1: torch.device("cuda:1" if torch.cuda.is_available() and self.num_gpus > 1 else "cpu")
        }
        
        # Support pour modèles multi-GPU (répartis sur les 2 GPUs)
        # Utilise l'ID spécial -1 pour identifier un modèle multi-GPU
        self.multi_gpu_model: Optional[Any] = None
        self.multi_gpu_tokenizer: Optional[Any] = None
        self.multi_gpu_model_name: Optional[str] = None
        self.multi_gpu_access_token: Optional[str] = None
        
    def _setup_miopen_cache(self, cache_size: int):
        """Configure le cache MIOpen pour ROCm."""
        os.environ["MIOPEN_USER_DB_PATH"] = os.path.expanduser("~/.cache/miopen")
        os.environ["MIOPEN_CACHE_DIR"] = os.path.expanduser("~/.cache/miopen")
        os.environ["MIOPEN_CACHE_MAX_SIZE"] = str(cache_size)
        os.environ["MIOPEN_ENABLE_CACHE"] = "1"
        logger.info(f"Cache MIOpen configuré: {cache_size} Mo dans {os.environ['MIOPEN_CACHE_DIR']}")
    
    def load_model(self, model_name: str, gpu_id: int = 0, **model_kwargs) -> tuple[bool, Optional[str]]:
        """
        Charge un modèle LLM sur le GPU spécifié.
        
        Args:
            model_name: Nom ou chemin du modèle Hugging Face
            gpu_id: ID du GPU (0 ou 1) sur lequel charger le modèle
            **model_kwargs: Arguments additionnels pour le chargement du modèle
                (ex: dtype, device_map, attn_implementation, etc.)
        
        Returns:
            (success: bool, access_token: Optional[str]) - True et le token si succès, False et None sinon
        """
        if gpu_id not in [0, 1]:
            logger.error(f"GPU ID invalide: {gpu_id}. Doit être 0 ou 1.")
            return False, None
        
        # Vérifier qu'un modèle multi-GPU n'est pas déjà chargé
        if self.multi_gpu_model is not None:
            logger.warning("Un modèle multi-GPU est déjà chargé. Déchargez-le d'abord pour charger un modèle sur un GPU individuel.")
            return False, None
        
        if self.models[gpu_id] is not None:
            logger.warning(f"Un modèle est déjà chargé sur GPU {gpu_id}. Déchargez-le d'abord.")
            return False, None
        
        if gpu_id >= self.num_gpus:
            logger.error(f"GPU {gpu_id} n'est pas disponible (seulement {self.num_gpus} GPU(s) détecté(s)).")
            return False, None
        
        device = self.devices[gpu_id]
        logger.info(f"Chargement du modèle '{model_name}' sur {device}...")
        
        try:
            # Chargement du tokenizer
            logger.info(f"Chargement du tokenizer pour '{model_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configuration spécifique pour les modèles Qwen
            model_name_lower = model_name.lower()
            if "qwen" in model_name_lower:
                # Pour Qwen, s'assurer que le pad_token est configuré correctement
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                # Qwen utilise parfois im_end comme pad_token
                if hasattr(tokenizer, 'im_end_id') and tokenizer.im_end_id is not None:
                    logger.debug(f"Qwen tokenizer configuré avec im_end_id: {tokenizer.im_end_id}")
                logger.info(f"Tokenizer Qwen configuré - pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")
            elif tokenizer.pad_token is None:
                # Pour les autres modèles, utiliser eos_token comme pad_token par défaut
                tokenizer.pad_token = tokenizer.eos_token
            
            # Chargement du modèle
            logger.info(f"Chargement du modèle sur {device}...")
            
            # Arguments par défaut pour le chargement
            load_kwargs = {
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                **model_kwargs
            }
            
            # Si device_map n'est pas explicitement fourni, on charge d'abord sur CPU puis on déplace
            # Cela évite les problèmes avec device_map sur certaines versions/configurations
            if "device_map" not in model_kwargs:
                # Charger sur CPU d'abord (plus compatible)
                load_kwargs["device_map"] = None
                load_kwargs["low_cpu_mem_usage"] = True
            
            # Désactiver SDPA attention pour ROCm multi-GPU (évite les problèmes de performance)
            # Utiliser "eager" pour forcer le backend attention standard
            if "attn_implementation" not in model_kwargs and torch.cuda.is_available():
                # Vérifier si on est sur ROCm (HIP)
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    load_kwargs["attn_implementation"] = "eager"
                    logger.info("SDPA attention désactivée pour ROCm (utilisation du backend 'eager')")
            
            # Chargement du modèle
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Déplacer le modèle sur le GPU spécifié manuellement
            model = model.to(device)
            model.eval()
            
            # Stockage
            self.models[gpu_id] = model
            self.tokenizers[gpu_id] = tokenizer
            self.model_names[gpu_id] = model_name
            
            # Génération d'un token d'accès unique pour ce GPU
            access_token = secrets.token_urlsafe(32)  # Token sécurisé de 32 bytes encodé en URL-safe
            self.access_tokens[gpu_id] = access_token
            
            logger.info(f"Modèle '{model_name}' chargé avec succès sur GPU {gpu_id}")
            logger.debug(f"Token d'accès généré pour GPU {gpu_id}: {access_token[:16]}...")
            
            # Vérification de l'emplacement du modèle
            if hasattr(model, 'device'):
                logger.info(f"Modèle placé sur: {model.device}")
            else:
                # Vérifier le premier paramètre
                first_param = next(model.parameters())
                logger.info(f"Modèle placé sur: {first_param.device}")
            
            return True, access_token
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Erreur lors du chargement du modèle '{model_name}': {error_msg}", exc_info=True)
            
            # Détection d'erreurs communes avec messages plus clairs
            if "GGUF" in error_msg or ".gguf" in error_msg.lower():
                logger.error("Modèle GGUF détecté - incompatible avec transformers. Utilisez llama.cpp ou un autre loader.")
            elif "No such file" in error_msg or "not found" in error_msg.lower():
                logger.error("Fichiers de modèle introuvables - vérifiez que le modèle est bien téléchargé.")
            elif "out of memory" in error_msg.lower() or "CUDA out of memory" in error_msg:
                logger.error("Mémoire GPU insuffisante - essayez un modèle plus petit ou libérez de l'espace.")
            
            # Nettoyage en cas d'erreur
            self.models[gpu_id] = None
            self.tokenizers[gpu_id] = None
            self.model_names[gpu_id] = None
            self.access_tokens[gpu_id] = None
            return False, None
    
    def unload_model(self, gpu_id: int, access_token: Optional[str] = None) -> tuple[bool, str]:
        """
        Décharge un modèle du GPU spécifié et nettoie la mémoire.
        
        Args:
            gpu_id: ID du GPU (0 ou 1) dont décharger le modèle
            access_token: Token d'accès requis pour décharger le modèle
        
        Returns:
            (success: bool, message: str) - True et message de succès, ou False et message d'erreur
        """
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
            return False, "Token d'accès invalide. Vous n'avez pas les permissions pour décharger ce modèle."
        
        model_name = self.model_names[gpu_id]
        logger.info(f"Déchargement du modèle '{model_name}' du GPU {gpu_id}...")
        
        try:
            # Suppression du modèle et du tokenizer
            del self.models[gpu_id]
            del self.tokenizers[gpu_id]
            self.models[gpu_id] = None
            self.tokenizers[gpu_id] = None
            self.model_names[gpu_id] = None
            # Suppression du token d'accès
            self.access_tokens[gpu_id] = None
            
            # Nettoyage de la mémoire Python
            gc.collect()
            
            # Nettoyage de la mémoire GPU
            if torch.cuda.is_available() and gpu_id < self.num_gpus:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device=self.devices[gpu_id])
                logger.info(f"Mémoire GPU {gpu_id} libérée.")
            
            logger.info(f"Modèle '{model_name}' déchargé avec succès.")
            return True, f"Modèle '{model_name}' déchargé avec succès du GPU {gpu_id}."
            
        except Exception as e:
            logger.error(f"Erreur lors du déchargement du modèle: {str(e)}", exc_info=True)
            return False, f"Erreur lors du déchargement: {str(e)}"
    
    def load_model_multi_gpu(self, model_name: str, **model_kwargs) -> tuple[bool, Optional[str]]:
        """
        Charge un modèle LLM réparti sur les 2 GPUs (multi-GPU).
        Utile pour charger des modèles plus grands que la mémoire d'un seul GPU.
        
        Args:
            model_name: Nom ou chemin du modèle Hugging Face
            **model_kwargs: Arguments additionnels pour le chargement du modèle
                (ex: dtype, attn_implementation, max_memory, etc.)
        
        Returns:
            (success: bool, access_token: Optional[str]) - True et le token si succès, False et None sinon
        """
        # Vérifier qu'on a au moins 2 GPUs disponibles
        if self.num_gpus < 2:
            logger.error(f"Multi-GPU nécessite au moins 2 GPUs, seulement {self.num_gpus} détecté(s).")
            return False, None
        
        # Vérifier qu'aucun modèle n'est déjà chargé
        if self.models[0] is not None or self.models[1] is not None:
            logger.warning("Des modèles sont déjà chargés sur les GPUs individuels. Déchargez-les d'abord pour utiliser multi-GPU.")
            return False, None
        
        # Vérifier qu'un modèle multi-GPU n'est pas déjà chargé
        if self.multi_gpu_model is not None:
            logger.warning("Un modèle multi-GPU est déjà chargé. Déchargez-le d'abord.")
            return False, None
        
        logger.info(f"Chargement du modèle '{model_name}' en mode multi-GPU (répartition sur GPU 0 et GPU 1)...")
        
        try:
            # Chargement du tokenizer
            logger.info(f"Chargement du tokenizer pour '{model_name}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configuration spécifique pour les modèles Qwen
            model_name_lower = model_name.lower()
            if "qwen" in model_name_lower:
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                if hasattr(tokenizer, 'im_end_id') and tokenizer.im_end_id is not None:
                    logger.debug(f"Qwen tokenizer configuré avec im_end_id: {tokenizer.im_end_id}")
                logger.info(f"Tokenizer Qwen configuré - pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")
            elif tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configuration pour le chargement multi-GPU
            load_kwargs = {
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                **model_kwargs
            }
            
            # Configuration du device_map pour répartir sur les 2 GPUs
            # Si device_map n'est pas fourni, utiliser "auto" qui répartit automatiquement
            if "device_map" not in model_kwargs:
                # "auto" répartit automatiquement les couches sur les GPUs disponibles
                # On peut aussi spécifier manuellement avec un dict: {0: "8GiB", 1: "8GiB"}
                load_kwargs["device_map"] = "auto"
                logger.info("Utilisation de device_map='auto' pour répartir automatiquement sur les GPUs")
                
                # Optionnel: limiter la mémoire par GPU (en Go)
                # Pour Vega 64 (8Go), on peut allouer environ 7.5Go par GPU pour laisser de la marge
                if "max_memory" not in model_kwargs:
                    max_memory = {0: "7.5GiB", 1: "7.5GiB"}
                    load_kwargs["max_memory"] = max_memory
                    logger.info(f"Limite de mémoire configurée: {max_memory}")
            
            # Désactiver SDPA attention pour ROCm multi-GPU
            if "attn_implementation" not in model_kwargs and torch.cuda.is_available():
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    load_kwargs["attn_implementation"] = "eager"
                    logger.info("SDPA attention désactivée pour ROCm multi-GPU (utilisation du backend 'eager')")
            
            # Activer low_cpu_mem_usage pour économiser la RAM système
            load_kwargs["low_cpu_mem_usage"] = True
            
            # Chargement du modèle avec répartition multi-GPU
            logger.info("Chargement du modèle avec répartition automatique sur les 2 GPUs...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Le modèle est maintenant réparti sur les GPUs via device_map
            model.eval()
            
            # Stockage
            self.multi_gpu_model = model
            self.multi_gpu_tokenizer = tokenizer
            self.multi_gpu_model_name = model_name
            
            # Génération d'un token d'accès unique
            access_token = secrets.token_urlsafe(32)
            self.multi_gpu_access_token = access_token
            
            logger.info(f"Modèle '{model_name}' chargé avec succès en mode multi-GPU (GPU 0 + GPU 1)")
            logger.debug(f"Token d'accès généré pour modèle multi-GPU: {access_token[:16]}...")
            
            # Afficher la répartition du modèle
            if hasattr(model, 'hf_device_map'):
                logger.info(f"Répartition du modèle: {model.hf_device_map}")
            
            # Vérifier où sont placés les paramètres
            param_devices = set()
            for name, param in model.named_parameters():
                if param.device.type == 'cuda':
                    param_devices.add(str(param.device))
            logger.info(f"Paramètres du modèle sur les devices: {sorted(param_devices)}")
            
            return True, access_token
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Erreur lors du chargement multi-GPU du modèle '{model_name}': {error_msg}", exc_info=True)
            
            # Détection d'erreurs communes
            if "out of memory" in error_msg.lower() or "CUDA out of memory" in error_msg:
                logger.error("Mémoire GPU insuffisante même en mode multi-GPU - le modèle est trop grand pour 2x8Go.")
            elif "GGUF" in error_msg or ".gguf" in error_msg.lower():
                logger.error("Modèle GGUF détecté - incompatible avec transformers.")
            
            # Nettoyage en cas d'erreur
            self.multi_gpu_model = None
            self.multi_gpu_tokenizer = None
            self.multi_gpu_model_name = None
            self.multi_gpu_access_token = None
            return False, None
    
    def unload_model_multi_gpu(self, access_token: Optional[str] = None) -> tuple[bool, str]:
        """
        Décharge un modèle multi-GPU et nettoie la mémoire des 2 GPUs.
        
        Args:
            access_token: Token d'accès requis pour décharger le modèle
        
        Returns:
            (success: bool, message: str) - True et message de succès, ou False et message d'erreur
        """
        if self.multi_gpu_model is None:
            return False, "Aucun modèle multi-GPU chargé."
        
        # Vérification du token d'accès
        if access_token is None:
            return False, "Token d'accès requis pour décharger le modèle multi-GPU."
        
        if self.multi_gpu_access_token is None:
            return False, "Aucun token d'accès associé au modèle multi-GPU."
        
        if self.multi_gpu_access_token != access_token:
            logger.warning("Tentative de déchargement multi-GPU avec un token invalide")
            return False, "Token d'accès invalide. Vous n'avez pas les permissions pour décharger ce modèle."
        
        model_name = self.multi_gpu_model_name
        logger.info(f"Déchargement du modèle multi-GPU '{model_name}'...")
        
        try:
            # Suppression du modèle et du tokenizer
            del self.multi_gpu_model
            del self.multi_gpu_tokenizer
            self.multi_gpu_model = None
            self.multi_gpu_tokenizer = None
            self.multi_gpu_model_name = None
            self.multi_gpu_access_token = None
            
            # Nettoyage de la mémoire Python
            gc.collect()
            
            # Nettoyage de la mémoire des 2 GPUs
            if torch.cuda.is_available():
                for gpu_id in [0, 1]:
                    if gpu_id < self.num_gpus:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(device=self.devices[gpu_id])
                logger.info("Mémoire GPU 0 et GPU 1 libérée.")
            
            logger.info(f"Modèle multi-GPU '{model_name}' déchargé avec succès.")
            return True, f"Modèle multi-GPU '{model_name}' déchargé avec succès."
            
        except Exception as e:
            logger.error(f"Erreur lors du déchargement du modèle multi-GPU: {str(e)}", exc_info=True)
            return False, f"Erreur lors du déchargement: {str(e)}"
    
    def generate(
        self,
        prompt: str,
        gpu_id: int = 0,
        max_length: int = 512,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generation_kwargs
    ) -> Optional[str]:
        """
        Génère une réponse à partir d'un prompt en utilisant le modèle sur le GPU spécifié.
        
        Args:
            prompt: Texte d'entrée (prompt)
            gpu_id: ID du GPU (0, 1, ou -1 pour modèle multi-GPU) à utiliser pour l'inférence
            max_length: Longueur maximale totale (prompt + génération)
            max_new_tokens: Nombre maximum de nouveaux tokens à générer (prioritaire sur max_length)
            temperature: Température pour la génération (0.0-1.0)
            top_p: Top-p (nucleus) sampling
            do_sample: Activer l'échantillonnage
            **generation_kwargs: Arguments additionnels pour la génération
        
        Returns:
            Texte généré ou None en cas d'erreur
        """
        # Vérifier si on utilise le modèle multi-GPU
        use_multi_gpu = (gpu_id == -1)
        
        if use_multi_gpu:
            if self.multi_gpu_model is None:
                logger.error("Aucun modèle multi-GPU chargé. Utilisez load_model_multi_gpu() d'abord.")
                return None
            model = self.multi_gpu_model
            tokenizer = self.multi_gpu_tokenizer
            model_name = self.multi_gpu_model_name
            # Pour multi-GPU, les inputs seront automatiquement gérés par le device_map
            device = None  # Pas de device unique pour multi-GPU
        else:
            if gpu_id not in [0, 1]:
                logger.error(f"GPU ID invalide: {gpu_id}. Doit être 0, 1, ou -1 (multi-GPU).")
                return None
            
            if self.models[gpu_id] is None:
                logger.error(f"Aucun modèle chargé sur GPU {gpu_id}.")
                return None
            
            model = self.models[gpu_id]
            tokenizer = self.tokenizers[gpu_id]
            device = self.devices[gpu_id]
            model_name = self.model_names[gpu_id]
        
        try:
            # Détection du type de modèle pour un traitement spécifique
            model_name_lower = model_name.lower() if model_name else ""
            is_qwen = "qwen" in model_name_lower
            use_chat_template = False
            formatted_prompt = prompt
            original_prompt = prompt  # Conserver le prompt original pour le décodage
            
            # Pour les modèles de chat, essayer d'utiliser apply_chat_template si disponible
            # Cela formate correctement les prompts pour les modèles conversationnels
            if hasattr(tokenizer, 'apply_chat_template') and callable(getattr(tokenizer, 'apply_chat_template', None)):
                try:
                    # Si le prompt contient plusieurs lignes (messages séparés), essayer de les parser
                    # Sinon, traiter comme un seul message utilisateur
                    prompt_lines = [line.strip() for line in prompt.split('\n') if line.strip()]
                    if len(prompt_lines) > 1:
                        # Plusieurs messages - essayer de les formater en format chat
                        messages = []
                        for line in prompt_lines:
                            messages.append({"role": "user", "content": line})
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    else:
                        # Un seul message
                        messages = [{"role": "user", "content": prompt.strip()}]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    use_chat_template = True
                    logger.debug(f"Prompt formaté avec apply_chat_template (modèle: {model_name}): {formatted_prompt[:200]}...")
                except Exception as e:
                    logger.warning(f"Erreur avec apply_chat_template pour {model_name}, utilisation du prompt brut: {e}", exc_info=True)
                    # Fallback : utiliser le prompt tel quel
                    formatted_prompt = prompt
                    use_chat_template = False
            
            # Logging supplémentaire pour le débogage
            if is_qwen:
                logger.debug(f"Traitement Qwen - use_chat_template: {use_chat_template}, prompt_length: {len(prompt)}, formatted_length: {len(formatted_prompt)}")
            
            # Tokenisation : pas de padding nécessaire pour une seule séquence de génération
            # Le padding est seulement utile pour le traitement par batch
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=False, truncation=True)
            
            # Pour Qwen, s'assurer que le tokenizer a les bons paramètres
            if is_qwen and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug("Token de padding configuré pour Qwen (utilise eos_token)")
            
            # Pour multi-GPU, placer les inputs sur le premier GPU (cuda:0)
            # Le modèle gérera automatiquement la répartition via device_map
            if use_multi_gpu:
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Génération
            with torch.no_grad():
                # S'assurer que la température est toujours > 0 pour éviter les problèmes numériques
                # Une température trop basse (< 0.01) peut causer des NaN/inf dans les probabilités
                safe_temperature = max(temperature, 0.01) if do_sample else temperature
                
                generation_config = {
                    "max_length": max_length,
                    "temperature": safe_temperature,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "repetition_penalty": 1.1,  # Pénalité contre les répétitions (1.0 = pas de pénalité, >1.0 = pénalise les répétitions)
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    **generation_kwargs
                }
                
                # Si max_new_tokens est spécifié, l'utiliser à la place de max_length
                if max_new_tokens is not None:
                    generation_config["max_new_tokens"] = max_new_tokens
                    generation_config.pop("max_length", None)
                
                # Pour éviter les problèmes de probabilités invalides, s'assurer que top_p est valide
                if do_sample and top_p <= 0:
                    logger.warning(f"top_p invalide ({top_p}), utilisation de 0.9 par défaut")
                    generation_config["top_p"] = 0.9
                
                # Si la température a été ajustée, logger l'avertissement
                if safe_temperature != temperature and do_sample:
                    logger.debug(f"Température ajustée de {temperature} à {safe_temperature} pour éviter les problèmes numériques")
                
                try:
                    device_info = "multi-GPU" if use_multi_gpu else str(device)
                    logger.debug(f"Début génération - input_ids shape: {inputs['input_ids'].shape}, device: {device_info}")
                    outputs = model.generate(**inputs, **generation_config)
                    logger.debug(f"Génération terminée - outputs shape: {outputs.shape}")
                except RuntimeError as e:
                    # Gestion spécifique de l'erreur de probabilités invalides
                    error_msg = str(e)
                    logger.error(f"RuntimeError lors de la génération pour {model_name}: {error_msg}")
                    if "probability tensor" in error_msg.lower() or "nan" in error_msg.lower() or "inf" in error_msg.lower():
                        logger.warning(f"Erreur de probabilités invalides détectée: {error_msg}")
                        # Réessayer avec des paramètres plus stables
                        logger.info("Réessai avec paramètres de génération plus stables...")
                        generation_config["temperature"] = max(safe_temperature, 0.5)  # Température minimale plus élevée
                        generation_config["top_p"] = min(top_p, 0.95)  # top_p légèrement réduit
                        # Ajouter top_k pour limiter le nombre de tokens candidats
                        if "top_k" not in generation_config:
                            generation_config["top_k"] = 50
                        logger.debug(f"Nouveaux paramètres: temp={generation_config['temperature']}, top_p={generation_config['top_p']}, top_k={generation_config.get('top_k')}")
                        outputs = model.generate(**inputs, **generation_config)
                    else:
                        # Autre erreur RuntimeError, la remonter
                        raise
                except Exception as e:
                    logger.error(f"Erreur inattendue lors de la génération pour {self.model_names[gpu_id]}: {str(e)}", exc_info=True)
                    raise
            
            # Décodage de la réponse
            # Toujours extraire uniquement les nouveaux tokens générés
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs[0].shape[0]
            
            # Vérifier qu'on a bien généré de nouveaux tokens
            if output_length <= input_length:
                logger.warning(f"Aucun nouveau token généré pour {model_name} - output_length={output_length}, input_length={input_length}")
                return ""
            
            generated_ids = outputs[0][input_length:]
            logger.debug(f"Tokens générés: {len(generated_ids)} tokens (input: {input_length}, output: {output_length})")
            
            # Décoder les nouveaux tokens uniquement
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            logger.debug(f"Texte décodé brut: {generated_text[:100]}...")
            
            # Nettoyer la réponse : retirer les tokens spéciaux de fin de conversation pour les modèles de chat
            # Tokens spéciaux spécifiques à Qwen et autres modèles
            generated_text = generated_text.strip()
            
            # Retirer les préfixes/suffixes communs des modèles de chat qui peuvent rester
            # Spécifiques à Qwen: <|im_start|>, <|im_end|>, <|endoftext|>
            chat_end_patterns = [
                '</s>', 
                '<|endoftext|>', 
                '<|end|>', 
                '<|im_end|>',
                '<|im_start|>',
                '\nUser:', 
                '\nAssistant:', 
                '\nSystem:',
                'User:',
                'Assistant:',
                'System:'
            ]
            
            # Retirer les patterns de fin
            for pattern in chat_end_patterns:
                # Retirer à la fin
                while generated_text.endswith(pattern):
                    generated_text = generated_text[:-len(pattern)].strip()
                # Retirer au début
                while generated_text.startswith(pattern):
                    generated_text = generated_text[len(pattern):].strip()
            
            # Pour Qwen spécifiquement, nettoyer les tokens de formatage qui peuvent rester
            if is_qwen:
                # Retirer les tokens de formatage Qwen qui peuvent apparaître
                qwen_patterns = [
                    '<|im_start|>assistant\n',
                    '<|im_end|>\n',
                    '\n<|im_end|>',
                    '<|im_start|>',
                ]
                for pattern in qwen_patterns:
                    generated_text = generated_text.replace(pattern, '')
                generated_text = generated_text.strip()
            
            logger.debug(f"Texte généré (après nettoyage): {generated_text[:200]}...")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}", exc_info=True)
            return None
    
    def get_model_status(self, gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Retourne le statut des modèles chargés.
        
        Args:
            gpu_id: ID du GPU spécifique (None pour tous les GPUs, -1 pour multi-GPU)
        
        Returns:
            Dictionnaire avec les informations sur les modèles
        """
        # Statut pour modèle multi-GPU
        if gpu_id == -1:
            return {
                "gpu_id": -1,
                "is_multi_gpu": True,
                "model_loaded": self.multi_gpu_model is not None,
                "model_name": self.multi_gpu_model_name,
                "devices": ["cuda:0", "cuda:1"] if self.num_gpus >= 2 else [],
                "has_access_token": self.multi_gpu_access_token is not None,
                "gpus_used": [0, 1] if self.multi_gpu_model is not None and self.num_gpus >= 2 else []
            }
        
        if gpu_id is not None:
            if gpu_id not in [0, 1]:
                return {"error": f"GPU ID invalide: {gpu_id}. Utilisez -1 pour multi-GPU."}
            
            return {
                "gpu_id": gpu_id,
                "model_loaded": self.models[gpu_id] is not None,
                "model_name": self.model_names[gpu_id],
                "device": str(self.devices[gpu_id]),
                "gpu_available": gpu_id < self.num_gpus
            }
        
        # Statut de tous les GPUs + multi-GPU
        status = {
            "total_gpus": self.num_gpus,
            "gpus": {},
            "multi_gpu": {
                "model_loaded": self.multi_gpu_model is not None,
                "model_name": self.multi_gpu_model_name,
                "devices": ["cuda:0", "cuda:1"] if self.num_gpus >= 2 else [],
                "has_access_token": self.multi_gpu_access_token is not None
            }
        }
        
        for gpu_id in [0, 1]:
            status["gpus"][gpu_id] = {
                "model_loaded": self.models[gpu_id] is not None,
                "model_name": self.model_names[gpu_id],
                "device": str(self.devices[gpu_id]),
                "gpu_available": gpu_id < self.num_gpus,
                "has_access_token": self.access_tokens[gpu_id] is not None
            }
            
            # Informations sur l'utilisation mémoire si disponible
            if torch.cuda.is_available() and gpu_id < self.num_gpus:
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                
                status["gpus"][gpu_id]["memory"] = {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "total_gb": round(memory_total, 2),
                    "free_gb": round(memory_total - memory_reserved, 2)
                }
        
        # Ajouter les informations mémoire pour le modèle multi-GPU
        if self.multi_gpu_model is not None and torch.cuda.is_available():
            status["multi_gpu"]["memory"] = {}
            for gpu_id in [0, 1]:
                if gpu_id < self.num_gpus:
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                    status["multi_gpu"]["memory"][f"gpu_{gpu_id}"] = {
                        "allocated_gb": round(memory_allocated, 2),
                        "reserved_gb": round(memory_reserved, 2),
                        "total_gb": round(memory_total, 2),
                        "free_gb": round(memory_total - memory_reserved, 2)
                    }
            
            # Afficher la répartition si disponible
            if hasattr(self.multi_gpu_model, 'hf_device_map'):
                status["multi_gpu"]["device_map"] = self.multi_gpu_model.hf_device_map
        
        return status
    
    def cleanup_all(self):
        """Décharge tous les modèles et nettoie toute la mémoire (admin/système uniquement)."""
        logger.info("Nettoyage complet de tous les modèles...")
        for gpu_id in [0, 1]:
            if self.models[gpu_id] is not None:
                # Pour cleanup_all, on utilise le token existant s'il y en a un
                token = self.access_tokens[gpu_id]
                if token:
                    self.unload_model(gpu_id, access_token=token)
                else:
                    # Fallback si pas de token (cas improbable mais géré)
                    logger.warning(f"Pas de token pour GPU {gpu_id}, déchargement forcé")
                    model_name = self.model_names[gpu_id]
                    del self.models[gpu_id]
                    del self.tokenizers[gpu_id]
                    self.models[gpu_id] = None
                    self.tokenizers[gpu_id] = None
                    self.model_names[gpu_id] = None
                    self.access_tokens[gpu_id] = None
                    gc.collect()
                    if torch.cuda.is_available() and gpu_id < self.num_gpus:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(device=self.devices[gpu_id])
                    logger.info(f"Modèle '{model_name}' déchargé du GPU {gpu_id}")
        
        # Nettoyage final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for gpu_id in range(self.num_gpus):
                torch.cuda.synchronize(device=self.devices[gpu_id])
        
        logger.info("Nettoyage complet terminé.")

