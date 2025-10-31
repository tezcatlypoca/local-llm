"""
Module pour gérer l'instance unique (singleton) du LLMManager.
Toutes les routes partagent cette même instance pour garantir la cohérence.
"""
from llm_manager import LLMManager

# Instance globale unique du gestionnaire LLM
_llm_manager_instance: LLMManager = None


def get_llm_manager() -> LLMManager:
    """
    Retourne l'instance unique (singleton) du gestionnaire LLM.
    
    Returns:
        Instance unique de LLMManager partagée entre toutes les routes
    """
    global _llm_manager_instance
    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()
        import logging
        logger = logging.getLogger(__name__)
        logger.info("LLMManager singleton initialisé")
    return _llm_manager_instance

