from src.utils.models_templates.qwen_template import format_qwen_messages
from src.utils.models_templates.mistral_template import format_mistral_messages
from typing import List, Dict, Callable, Optional

class TemplatesManager:
    """
    Gestionnaire des templates de formatage pour les différents modèles.
    """
    
    @classmethod
    def get_template(cls, model_name: str) -> Callable[[List[Dict[str, str]]], str]:
        """
        Récupère la fonction de formatage pour un modèle donné.
        
        Args:
            model_name: Nom du modèle (ex: 'qwen', 'mistral', ou nom complet)
        
        Returns:
            Fonction de formatage qui prend une liste de messages et retourne une chaîne formatée.
        
        Raises:
            ValueError: Si le modèle n'est pas supporté.
        """
        # Normaliser le nom du modèle (enlever les extensions, versions, etc.)
        normalized_name = TemplatesManager._normalize_model_name(model_name)
        
        if normalized_name not in cls._templates:
            raise ValueError(
                f"Template pour le modèle '{model_name}' (normalisé: '{normalized_name}') non trouvé. "
                f"Modèles supportés: {list(cls._templates.keys())}"
            )
        
        return cls._templates[normalized_name]
    
    @classmethod
    def format_messages(cls, model_name: str, messages: List[Dict[str, str]]) -> str:
        """
        Formate une liste de messages selon le template du modèle spécifié.
        
        Args:
            model_name: Nom du modèle (peut être un nom complet ou normalisé)
            messages: Liste de dictionnaires avec 'role' et 'content'
        
        Returns:
            Chaîne formatée selon le template du modèle
        """
        # Normaliser le nom du modèle
        normalized_name = cls._normalize_model_name(model_name)
        
        # Récupérer et appliquer le template
        template_func = cls.get_template(normalized_name)
        return template_func(messages)
    
    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """
        Normalise le nom du modèle pour le mapping avec les templates.
        
        Exemples:
            'mistral-7b-instruct-V0.2-Q4_K_M.gguf' -> 'mistral'
            'qwen2-7b-instruct' -> 'qwen'
            'qwen' -> 'qwen'
        
        Args:
            model_name: Nom complet ou partiel du modèle
        
        Returns:
            Nom normalisé du modèle
        """
        if not model_name:
            return model_name
        
        model_name_lower = model_name.lower()
        
        # Détecter Mistral (doit être en premier car 'mistral' peut être dans d'autres noms)
        if 'mistral' in model_name_lower:
            return 'mistral'
        
        # Détecter Qwen (gère qwen, qwen2, etc.)
        if 'qwen' in model_name_lower:
            return 'qwen'
        
        # Si aucun pattern ne correspond, retourner le nom tel quel (peut être déjà normalisé)
        return model_name_lower