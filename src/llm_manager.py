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
        
    def _setup_miopen_cache(self, cache_size: int):
        """Configure le cache MIOpen pour ROCm."""
        os.environ["MIOPEN_USER_DB_PATH"] = os.path.expanduser("~/.cache/miopen")
        os.environ["MIOPEN_CACHE_DIR"] = os.path.expanduser("~/.cache/miopen")
        os.environ["MIOPEN_CACHE_MAX_SIZE"] = str(cache_size)
        os.environ["MIOPEN_ENABLE_CACHE"] = "1"
        logger.info(f"Cache MIOpen configuré: {cache_size} Mo dans {os.environ['MIOPEN_CACHE_DIR']}")
    
    def load_model_multi_gpu(
        self, 
        model_name: str, 
        use_both_gpus: bool = True,
        **model_kwargs
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Charge un modèle LLM sur plusieurs GPU (expérimental).
        
        ⚠️ AVERTISSEMENT : Pour des modèles <8Go, le multi-GPU peut RÉDUIRE les performances
        à cause de la communication inter-GPU. Utilisez uniquement si :
        - Le modèle est trop grand pour un seul GPU
        - Vous testez expérimentalement
        
        Args:
            model_name: Nom ou chemin du modèle Hugging Face
            use_both_gpus: Si True, utilise les 2 GPU disponibles (défaut: True)
            **model_kwargs: Arguments additionnels pour le chargement du modèle
        
        Returns:
            (success: bool, access_token: Optional[str], warning: Optional[str])
        """
        if self.num_gpus < 2:
            return False, None, f"Au moins 2 GPU sont requis pour le multi-GPU. GPU disponibles: {self.num_gpus}"
        
        # Vérifier que les deux GPU sont libres
        if self.models[0] is not None or self.models[1] is not None:
            return False, None, "Les deux GPU doivent être libres pour le chargement multi-GPU. Déchargez les modèles existants d'abord."
        
        logger.warning("="*70)
        logger.warning("⚠️  CHARGEMENT MULTI-GPU EXPÉRIMENTAL ⚠️")
        logger.warning("Pour des modèles <8Go, les performances peuvent être INFÉRIEURES")
        logger.warning("à un chargement sur un seul GPU à cause de la communication inter-GPU.")
        logger.warning("="*70)
        
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
            
            # Configuration pour multi-GPU
            logger.info(f"Chargement du modèle '{model_name}' en mode multi-GPU...")
            
            load_kwargs = {
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if use_both_gpus else None,  # Répartition automatique
                "low_cpu_mem_usage": True,
                **model_kwargs
            }
            
            # Désactiver SDPA attention pour ROCm
            if "attn_implementation" not in model_kwargs and torch.cuda.is_available():
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    load_kwargs["attn_implementation"] = "eager"
                    logger.info("SDPA attention désactivée pour ROCm (utilisation du backend 'eager')")
            
            # Chargement du modèle avec répartition automatique
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            model.eval()
            
            # Stockage sur GPU 0 (le modèle est réparti mais on le référence via GPU 0)
            self.models[0] = model
            self.models[1] = model  # Même référence, modèle réparti
            self.tokenizers[0] = tokenizer
            self.tokenizers[1] = tokenizer
            self.model_names[0] = model_name
            self.model_names[1] = model_name
            
            # Génération d'un token d'accès unique
            access_token = secrets.token_urlsafe(32)
            self.access_tokens[0] = access_token
            self.access_tokens[1] = access_token  # Même token pour les deux GPU
            
            logger.info(f"Modèle '{model_name}' chargé avec succès en mode multi-GPU")
            
            # Vérifier la répartition
            if hasattr(model, 'hf_device_map'):
                logger.info(f"Répartition du modèle: {model.hf_device_map}")
            else:
                # Vérifier manuellement où sont les paramètres
                devices_used = set()
                for param in model.parameters():
                    devices_used.add(str(param.device))
                logger.info(f"Modèle réparti sur les devices: {devices_used}")
            
            warning_msg = (
                "⚠️ Mode multi-GPU activé. Pour des modèles <8Go, les performances peuvent être "
                "légèrement inférieures à un chargement sur un seul GPU. Utilisez l'inférence sur GPU 0."
            )
            
            return True, access_token, warning_msg
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Erreur lors du chargement multi-GPU du modèle '{model_name}': {error_msg}", exc_info=True)
            
            # Nettoyage en cas d'erreur
            self.models[0] = None
            self.models[1] = None
            self.tokenizers[0] = None
            self.tokenizers[1] = None
            self.model_names[0] = None
            self.model_names[1] = None
            self.access_tokens[0] = None
            self.access_tokens[1] = None
            
            if "out of memory" in error_msg.lower() or "CUDA out of memory" in error_msg:
                return False, None, "Mémoire GPU insuffisante - le modèle est peut-être trop grand même pour 2 GPU."
            
            return False, None, f"Erreur lors du chargement: {error_msg}"
    
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
    
    def unload_model_force(self, gpu_id: int, admin_password: str) -> tuple[bool, str]:
        """
        Décharge un modèle du GPU spécifié sans vérifier le token, uniquement avec le mot de passe admin.
        Cette méthode est utilisée par la route d'administration pour forcer le déchargement.
        
        Args:
            gpu_id: ID du GPU (0 ou 1) dont décharger le modèle
            admin_password: Mot de passe administrateur requis pour forcer le déchargement
        
        Returns:
            (success: bool, message: str) - True et message de succès, ou False et message d'erreur
        """
        ADMIN_PASSWORD = "admin69"
        
        if admin_password != ADMIN_PASSWORD:
            logger.warning(f"Tentative de déchargement forcé avec un mot de passe invalide sur GPU {gpu_id}")
            return False, "Mot de passe administrateur invalide."
        
        if gpu_id not in [0, 1]:
            return False, f"GPU ID invalide: {gpu_id}. Doit être 0 ou 1."
        
        if self.models[gpu_id] is None:
            return False, f"Aucun modèle chargé sur GPU {gpu_id}."
        
        model_name = self.model_names[gpu_id]
        logger.info(f"Déchargement forcé (admin) du modèle '{model_name}' du GPU {gpu_id}...")
        
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
            
            logger.info(f"Modèle '{model_name}' déchargé avec succès (admin).")
            return True, f"Modèle '{model_name}' déchargé avec succès du GPU {gpu_id} (admin)."
            
        except Exception as e:
            logger.error(f"Erreur lors du déchargement forcé du modèle: {str(e)}", exc_info=True)
            return False, f"Erreur lors du déchargement: {str(e)}"
    
    def unload_all_models(self, admin_password: str) -> Dict[str, Any]:
        """
        Décharge tous les modèles de tous les GPUs en utilisant le mot de passe admin.
        
        Args:
            admin_password: Mot de passe administrateur requis
        
        Returns:
            Dictionnaire avec les résultats du déchargement pour chaque GPU
        """
        results = {
            "gpu_0": {"success": False, "message": ""},
            "gpu_1": {"success": False, "message": ""}
        }
        
        # Décharger GPU 0
        if self.models[0] is not None:
            success, message = self.unload_model_force(0, admin_password)
            results["gpu_0"] = {"success": success, "message": message}
        else:
            results["gpu_0"] = {"success": True, "message": "Aucun modèle chargé sur GPU 0."}
        
        # Décharger GPU 1
        if self.models[1] is not None:
            success, message = self.unload_model_force(1, admin_password)
            results["gpu_1"] = {"success": success, "message": message}
        else:
            results["gpu_1"] = {"success": True, "message": "Aucun modèle chargé sur GPU 1."}
        
        return results
    
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
        
        Le prompt est utilisé tel quel, sans formatage ni template de contexte.
        La gestion du contexte et du formatage doit être effectuée par la surcouche API appelante.
        
        Args:
            prompt: Texte d'entrée (prompt brut, sans formatage)
            gpu_id: ID du GPU (0 ou 1) à utiliser pour l'inférence
            max_length: Longueur maximale totale (prompt + génération)
            max_new_tokens: Nombre maximum de nouveaux tokens à générer (prioritaire sur max_length)
            temperature: Température pour la génération (0.0-1.0)
            top_p: Top-p (nucleus) sampling
            do_sample: Activer l'échantillonnage
            **generation_kwargs: Arguments additionnels pour la génération
        
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
        tokenizer = self.tokenizers[gpu_id]
        device = self.devices[gpu_id]
        
        try:
            # Utiliser le prompt tel quel, sans formatage ni template
            # La surcouche API appelante doit gérer le formatage du contexte
            formatted_prompt = prompt.strip()
            
            # Tokenisation : pas de padding nécessaire pour une seule séquence de génération
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=False, truncation=True)
            logger.info(f"Input tokenisés - shape: {inputs['input_ids'].shape}, nombre de tokens: {inputs['input_ids'].shape[1]}")
            
            # S'assurer que le tokenizer a un pad_token si nécessaire
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Génération
            with torch.no_grad():
                # Configuration de sécurité pour éviter les problèmes de probabilités invalides (NaN/Inf) sur ROCm
                if do_sample:
                    # Température minimale pour éviter les problèmes numériques avec float16 sur ROCm
                    safe_temperature = max(temperature, 0.3)
                    # Top_k systématique pour limiter les candidats et éviter les valeurs extrêmes
                    safe_top_k = generation_kwargs.get("top_k", 50)
                    # Top_p légèrement réduit pour plus de stabilité
                    safe_top_p = min(top_p, 0.95)
                    # Réduction de la repetition_penalty pour éviter les problèmes numériques
                    safe_repetition_penalty = generation_kwargs.get("repetition_penalty", 1.1)
                    if safe_repetition_penalty > 1.2:
                        safe_repetition_penalty = 1.2
                    logger.debug(f"Paramètres ajustés - temp: {safe_temperature}, top_k: {safe_top_k}, top_p: {safe_top_p}, rep_penalty: {safe_repetition_penalty}")
                else:
                    # Pour le décodage greedy, utiliser les valeurs standard
                    safe_temperature = temperature
                    safe_top_k = generation_kwargs.get("top_k")
                    safe_top_p = top_p
                    safe_repetition_penalty = generation_kwargs.get("repetition_penalty", 1.1)
                
                generation_config = {
                    "max_length": max_length,
                    "temperature": safe_temperature,
                    "top_p": safe_top_p,
                    "do_sample": do_sample,
                    "repetition_penalty": safe_repetition_penalty,
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    **generation_kwargs
                }
                
                # Ajouter top_k si pas déjà présent et si disponible
                if do_sample and safe_top_k is not None and "top_k" not in generation_config:
                    generation_config["top_k"] = safe_top_k
                
                # Si max_new_tokens est spécifié, l'utiliser à la place de max_length
                if max_new_tokens is not None:
                    generation_config["max_new_tokens"] = max_new_tokens
                    generation_config.pop("max_length", None)
                
                # Pour éviter les problèmes de probabilités invalides, s'assurer que top_p est valide
                if do_sample and generation_config["top_p"] <= 0:
                    logger.warning(f"top_p invalide ({generation_config['top_p']}), utilisation de 0.9 par défaut")
                    generation_config["top_p"] = 0.9
                
                # Si la température a été ajustée, logger l'avertissement
                if safe_temperature != temperature and do_sample:
                    logger.debug(f"Température ajustée de {temperature} à {safe_temperature} pour éviter les problèmes numériques")
                
                try:
                    logger.debug(f"Début génération - input_ids shape: {inputs['input_ids'].shape}, device: {device}")
                    outputs = model.generate(**inputs, **generation_config)
                    logger.debug(f"Génération terminée - outputs shape: {outputs.shape}")
                except RuntimeError as e:
                    # Gestion spécifique de l'erreur de probabilités invalides
                    error_msg = str(e)
                    logger.error(f"RuntimeError lors de la génération pour {self.model_names[gpu_id]}: {error_msg}")
                    if "probability tensor" in error_msg.lower() or "nan" in error_msg.lower() or "inf" in error_msg.lower():
                        logger.warning(f"Erreur de probabilités invalides détectée: {error_msg}")
                        # Stratégie de réessai progressive
                        logger.info("Réessai avec paramètres de génération plus stables...")
                        generation_config["temperature"] = max(safe_temperature, 0.7)
                        generation_config["top_p"] = 0.9
                        generation_config["top_k"] = 40
                        generation_config["repetition_penalty"] = 1.1
                        logger.debug(f"Paramètres ajustés: temp={generation_config['temperature']}, top_p={generation_config['top_p']}, top_k={generation_config.get('top_k')}")
                        
                        try:
                            outputs = model.generate(**inputs, **generation_config)
                            logger.info("Réessai réussi avec paramètres ajustés")
                        except RuntimeError as e2:
                            error_msg2 = str(e2)
                            logger.warning(f"Deuxième tentative échouée: {error_msg2}")
                            # Dernière tentative: désactiver l'échantillonnage (greedy decoding)
                            logger.info("Dernière tentative avec décodage greedy (do_sample=False)...")
                            generation_config["do_sample"] = False
                            generation_config.pop("temperature", None)
                            generation_config.pop("top_p", None)
                            generation_config.pop("top_k", None)
                            try:
                                outputs = model.generate(**inputs, **generation_config)
                                logger.info("Réessai réussi avec décodage greedy")
                            except RuntimeError as e3:
                                logger.error(f"Toutes les tentatives ont échoué: {str(e3)}")
                                raise
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
                logger.warning(f"Aucun nouveau token généré pour {self.model_names[gpu_id]} - output_length={output_length}, input_length={input_length}")
                return ""
            
            generated_ids = outputs[0][input_length:]
            logger.debug(f"Tokens générés: {len(generated_ids)} tokens (input: {input_length}, output: {output_length})")
            
            # Décoder les nouveaux tokens uniquement
            # Utiliser skip_special_tokens=True pour retirer les tokens spéciaux du tokenizer
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            logger.info(f"Texte généré (longueur: {len(generated_text)}): {generated_text[:500]}")
            
            # Vérifier si la réponse contient uniquement des caractères répétitifs (signe de problème)
            if generated_text.strip():
                unique_chars = set(generated_text.strip())
                if len(unique_chars) == 1 and len(generated_text.strip()) > 10:
                    logger.warning(f"Réponse suspecte détectée: uniquement le caractère '{generated_text.strip()[0]}' répété {len(generated_text.strip())} fois")
            
            # Retourner le texte tel quel, sans nettoyage spécifique
            # La surcouche API appelante gère le formatage et le nettoyage si nécessaire
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}", exc_info=True)
            return None
    
    def get_model_status(self, gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Retourne le statut des modèles chargés.
        
        Args:
            gpu_id: ID du GPU spécifique (None pour tous les GPUs)
        
        Returns:
            Dictionnaire avec les informations sur les modèles
        """
        if gpu_id is not None:
            if gpu_id not in [0, 1]:
                return {"error": f"GPU ID invalide: {gpu_id}"}
            
            return {
                "gpu_id": gpu_id,
                "model_loaded": self.models[gpu_id] is not None,
                "model_name": self.model_names[gpu_id],
                "device": str(self.devices[gpu_id]),
                "gpu_available": gpu_id < self.num_gpus
            }
        
        # Statut de tous les GPUs
        status = {
            "total_gpus": self.num_gpus,
            "gpus": {}
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

