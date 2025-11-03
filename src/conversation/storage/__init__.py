"""
Module de persistance des conversations.
"""
from .base import StorageBackend
from .sqlite_storage import SQLiteStorage

__all__ = [
    "StorageBackend",
    "SQLiteStorage",
]

