"""Edison Common Utilities

This module provides common utilities for the Edison project.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import os
from typing import Optional
from uuid import uuid4
from ..errors import StorageError


def ensure_dir(path: str) -> None:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Raises:
        StorageError: If directory cannot be created or accessed
    """
    if ":" in path and not path.startswith("/dev"):
        raise StorageError(f"Invalid path containing colon: {path}")

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except (OSError, PermissionError) as e:
        raise StorageError(f"Failed to create directory {path}: {e}")


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to be safe for filesystem operations.

    Args:
        filename: The filename to sanitize

    Returns:
        A sanitized version of the filename
    """
    # Remove or replace illegal characters
    illegal = '<>:"/\\|?*'
    for char in illegal:
        filename = filename.replace(char, "_")
    return filename.strip()


def get_document_id(title: str, timestamp: Optional[str] = None) -> str:
    """Generate a document ID from a title.

    Args:
        title: The document title
        timestamp: Optional timestamp to append

    Returns:
        A filesystem-safe document ID
    """
    doc_id = sanitize_filename(title.lower().replace(" ", "_"))
    if timestamp:
        doc_id = f"{doc_id}_{timestamp}"
    return doc_id


def generate_document_id() -> str:
    """Generate a document ID based on a query.

    Args:
        query: The query string

    Returns:
        A sanitized document ID
    """
    return uuid4()
