"""Edison Common Utilities

This module provides common utilities for the Edison project.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import os
from typing import Optional


def ensure_dir(path: str) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


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
