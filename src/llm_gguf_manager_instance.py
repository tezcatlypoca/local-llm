"""
Module pour gérer l'instance unique (singleton) du LLMGGUFManager.
"""
from llm_gguf_manager import LLMGGUFManager

# Instance globale unique du gestionnaire GGUF
_llm_gguf_manager_instance: LLMGGUFManager = None


def get_gguf_manager() -> LLMGGUFManager:
    """
    Retourne l'instance unique (singleton) du gestionnaire GGUF.
    
    Returns:
        Instance unique de LLMGGUFManager partagée entre toutes les routes
    """
    global _llm_gguf_manager_instance
    if _llm_gguf_manager_instance is None:
        _llm_gguf_manager_instance = LLMGGUFManager()
        import logging
        logger = logging.getLogger(__name__)
        logger.info("LLMGGUFManager singleton initialisé")
    return _llm_gguf_manager_instance

