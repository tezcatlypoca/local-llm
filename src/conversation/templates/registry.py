"""
Registre centralisé des templates.
"""
from typing import Dict, Type, Optional
from .base import Template
from .tinyllama import TinyLlamaChatTemplate
from .qwen25 import Qwen25Template
from .mistral import MistralInstructTemplate
from ..exceptions import TemplateNotFoundError


class TemplateRegistry:
    """
    Registre centralisé des templates.
    
    Permet de récupérer automatiquement le bon template
    en fonction du nom du modèle.
    """
    
    # Mapping nom modèle -> classe template (recherche exacte)
    _template_classes: Dict[str, Type[Template]] = {
        # TinyLlama
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": TinyLlamaChatTemplate,
        "tinyllama": TinyLlamaChatTemplate,
        
        # Qwen2.5
        "Qwen/Qwen2.5-7B": Qwen25Template,
        "Qwen/Qwen2.5-7B-Instruct": Qwen25Template,
        "qwen2.5": Qwen25Template,
        "qwen": Qwen25Template,
        
        # Mistral 7B Instruct
        "mistralai/Mistral-7B-Instruct-v0.1": MistralInstructTemplate,
        "mistralai/Mistral-7B-Instruct-v0.2": MistralInstructTemplate,
        "mistralai/Mistral-7B-Instruct-v0.3": MistralInstructTemplate,
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": MistralInstructTemplate,
        "mistral-7b-instruct": MistralInstructTemplate,
        "mistral": MistralInstructTemplate,
    }
    
    # Patterns pour détecter le type de modèle (recherche par pattern)
    _model_patterns: Dict[str, Type[Template]] = {
        "tinyllama": TinyLlamaChatTemplate,
        "qwen2.5": Qwen25Template,
        "qwen": Qwen25Template,  # Plus générique après qwen2.5
        "mistral-7b-instruct": MistralInstructTemplate,
        "mistral": MistralInstructTemplate,
    }
    
    @classmethod
    def get_template(cls, model_name: str) -> Template:
        """
        Récupère le template approprié pour un modèle.
        
        Args:
            model_name: Nom du modèle (ex: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        Returns:
            Instance du template approprié
        
        Raises:
            TemplateNotFoundError: Si aucun template n'est trouvé
        """
        # Normaliser le nom du modèle (enlever les espaces, mettre en minuscule)
        normalized_name = model_name.strip().lower()
        
        # Recherche exacte (sensible à la casse pour le mapping original)
        if model_name in cls._template_classes:
            template_class = cls._template_classes[model_name]
            return template_class(model_name)
        
        # Recherche exacte normalisée
        if normalized_name in cls._template_classes:
            template_class = cls._template_classes[normalized_name]
            return template_class(model_name)
        
        # Recherche par pattern dans le nom normalisé
        for pattern, template_class in cls._model_patterns.items():
            if pattern in normalized_name:
                return template_class(model_name)
        
        # Si aucune correspondance, on lève une exception
        raise TemplateNotFoundError(model_name)
    
    @classmethod
    def register_template(cls, model_name: str, template_class: Type[Template]):
        """
        Enregistre un nouveau template personnalisé.
        
        Args:
            model_name: Nom du modèle à associer
            template_class: Classe du template à utiliser
        """
        cls._template_classes[model_name] = template_class
    
    @classmethod
    def list_supported_models(cls) -> list:
        """
        Liste les modèles supportés.
        
        Returns:
            Liste des noms de modèles supportés
        """
        return list(cls._template_classes.keys())

