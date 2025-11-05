"""
Module de persistance des conversations.
"""
from .base import StorageBackend
from .sqlite_storage import SQLiteStorage
from .json_storage import JSONStorage

__all__ = [
    "StorageBackend",
    "SQLiteStorage",
    "JSONStorage",
]

