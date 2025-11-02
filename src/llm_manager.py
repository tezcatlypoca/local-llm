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
            if tokenizer.pad_token is None:
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
                    logger.debug(f"Prompt formaté avec apply_chat_template: {formatted_prompt[:200]}...")
                    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
                except Exception as e:
                    logger.debug(f"Erreur avec apply_chat_template, utilisation du prompt brut: {e}")
                    # Fallback : utiliser le prompt tel quel
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            else:
                # Tokenisation standard pour les modèles non-chat
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
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
                    outputs = model.generate(**inputs, **generation_config)
                except RuntimeError as e:
                    # Gestion spécifique de l'erreur de probabilités invalides
                    error_msg = str(e)
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
            
            # Décodage de la réponse
            # Si on veut seulement les nouveaux tokens générés
            if max_new_tokens is not None:
                generated_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Retirer le prompt de la réponse complète
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
            
            # Nettoyer la réponse : retirer les tokens spéciaux de fin de conversation pour les modèles de chat
            # (ex: </s>, <|endoftext|>, etc.)
            generated_text = generated_text.strip()
            
            # Retirer les préfixes/suffixes communs des modèles de chat qui peuvent rester
            chat_end_patterns = [
                '</s>', '<|endoftext|>', '<|end|>', '<|im_end|>',
                '\nUser:', '\nAssistant:', '\nSystem:'
            ]
            for pattern in chat_end_patterns:
                if generated_text.endswith(pattern):
                    generated_text = generated_text[:-len(pattern)].strip()
                if generated_text.startswith(pattern):
                    generated_text = generated_text[len(pattern):].strip()
            
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

