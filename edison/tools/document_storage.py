"""Document storage interface for Edison document writer.

This module provides storage implementations for document persistence operations,
allowing documents to be saved and retrieved between sessions. It handles
JSON-based storage of document content, metadata, and section information.

Typical usage example:
    storage = DocumentStorage("/path/to/storage")
    document = storage.load_document("doc_id")
    storage.save_document("doc_id", document_content)
    all_docs = storage.list_documents()

Note:
    All storage operations require valid document IDs and proper storage directory
    configuration. Documents are stored as JSON files with .json extension.
"""

import json
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from .document_tools import DocumentContent, DocumentSection
from ..errors import StorageError, StorageIOError


class DocumentStorage:
    """Interface for document persistence operations.

    This class handles the storage and retrieval of document content to/from
    the filesystem. Documents are stored as JSON files containing sections,
    metadata, and version information.

    Attributes:
        storage_dir (Path): Base directory path for document storage
    """

    def __init__(self, storage_dir: str):
        """Initialize storage with directory path.

        Creates the storage directory if it doesn't exist and prepares
        the system for document storage operations.

        Args:
            storage_dir: Base directory path for document storage

        Raises:
            StorageError: If storage directory cannot be created or accessed
        """
        self.storage_dir = Path(storage_dir)
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise StorageError(f"Failed to initialize storage directory: {e}")

    def _get_doc_path(self, doc_id: str) -> Path:
        """Get full path for document storage.

        Constructs the complete filesystem path for a document based on
        its ID and the storage directory.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            Path object representing the document's storage location
        """
        return self.storage_dir / f"{doc_id}.json"

    def save_document(self, doc_id: str, document: DocumentContent) -> None:
        """Saves document content to storage.

        Persists the complete document content including sections, metadata,
        and version information to a JSON file in the storage directory.

        Args:
            doc_id: Unique identifier for the document
            document: Document content object to be saved

        Raises:
            StorageIOError: If document cannot be saved due to IO errors
            or serialization issues
        """
        doc_path = self._get_doc_path(doc_id)

        try:
            # Convert to serializable format
            doc_dict = {
                "sections": {
                    id: {
                        "title": section.title,
                        "content": section.content,
                        "last_modified": section.last_modified.isoformat(),
                        "version": section.version,
                        "context_tokens": section.context_tokens,
                    }
                    for id, section in document.sections.items()
                },
                "metadata": document.metadata,
                "created_at": document.created_at.isoformat(),
                "last_modified": document.last_modified.isoformat(),
                "version": document.version,
            }

            with doc_path.open("w") as f:
                json.dump(doc_dict, f, indent=2)
        except Exception as e:
            raise StorageIOError(f"Failed to save document {doc_id}: {e}")

    def load_document(self, doc_id: str) -> Optional[DocumentContent]:
        """Loads document content from storage.

        Retrieves and deserializes document content from storage, reconstructing
        the DocumentContent object with all sections and metadata.

        Args:
            doc_id: Unique identifier for the document to load

        Returns:
            DocumentContent object if document exists, None otherwise

        Raises:
            StorageIOError: If document exists but cannot be loaded due to
            IO errors or invalid format
        """
        doc_path = self._get_doc_path(doc_id)

        if not doc_path.exists():
            return None

        try:
            with doc_path.open("r") as f:
                doc_dict = json.load(f)

            # Convert back to DocumentContent
            sections = {
                id: DocumentSection(
                    title=section["title"],
                    content=section["content"],
                    last_modified=datetime.fromisoformat(section["last_modified"]),
                    version=section["version"],
                    context_tokens=section["context_tokens"],
                )
                for id, section in doc_dict["sections"].items()
            }

            return DocumentContent(
                sections=sections,
                metadata=doc_dict["metadata"],
                created_at=datetime.fromisoformat(doc_dict["created_at"]),
                last_modified=datetime.fromisoformat(doc_dict["last_modified"]),
                version=doc_dict["version"],
            )
        except Exception as e:
            raise StorageIOError(f"Failed to load document {doc_id}: {e}")

    def list_documents(self) -> Dict[str, Dict[str, str]]:
        """List all available documents with metadata.

        Scans the storage directory for document files and extracts their
        metadata information.

        Returns:
            Dictionary mapping document IDs to their metadata dictionaries

        Raises:
            StorageIOError: If storage directory cannot be read or document
            metadata extraction fails
        """
        documents = {}

        try:
            for doc_path in self.storage_dir.glob("*.json"):
                doc_id = doc_path.stem

                try:
                    with doc_path.open("r") as f:
                        doc_dict = json.load(f)
                        documents[doc_id] = doc_dict["metadata"]
                except Exception as e:
                    # Skip invalid documents but continue processing others
                    continue

            return documents
        except Exception as e:
            raise StorageIOError(f"Failed to list documents: {e}")
