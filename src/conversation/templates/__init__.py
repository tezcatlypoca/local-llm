"""
Templates de formatage pour différents modèles LLM.
"""
from .base import Template
from .tinyllama import TinyLlamaChatTemplate
from .qwen25 import Qwen25Template
from .mistral import MistralInstructTemplate
from .registry import TemplateRegistry

__all__ = [
    "Template",
    "TinyLlamaChatTemplate",
    "Qwen25Template",
    "MistralInstructTemplate",
    "TemplateRegistry",
]

