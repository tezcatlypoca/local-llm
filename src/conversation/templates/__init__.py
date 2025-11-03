"""
Templates de formatage pour différents modèles LLM.
"""
from .base import Template
from .tinyllama import TinyLlamaChatTemplate
from .finbert import FinBertTemplate
from .qwen25 import Qwen25Template
from .registry import TemplateRegistry

__all__ = [
    "Template",
    "TinyLlamaChatTemplate",
    "FinBertTemplate",
    "Qwen25Template",
    "TemplateRegistry",
]

