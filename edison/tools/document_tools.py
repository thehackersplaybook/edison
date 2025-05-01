"""Document management tools for Edison agents.

This module provides tools for managing documents, including versioning,
content organization, and automatic restructuring.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from ..errors import DocumentNotFoundError
from ..models import (
    DocumentContent,
    DocumentSection,
)
from .document_storage import DocumentStorage
from ..common.utils import ensure_dir


class DocumentWriterTool:
    """A tool for managing document content with versioning and organization."""

    def __init__(self, storage_dir: str = "documents"):
        """Initialize the document writer tool.

        Args:
            storage_dir: Directory to store documents
        """
        self.storage_dir = storage_dir
        ensure_dir(storage_dir)
        self.storage = DocumentStorage(storage_dir)
        self.documents: Dict[str, DocumentContent] = {}
        self._load_existing_documents()

    def _load_existing_documents(self) -> None:
        """Load any existing documents from storage."""
        for doc_id in self.storage.list_documents():
            if doc := self.storage.load_document(doc_id):
                self.documents[doc_id] = doc

    def create_document(self, doc_id: str, metadata: Dict[str, str]) -> DocumentContent:
        """Creates a new empty document."""
        now = datetime.now()
        doc = DocumentContent(
            sections={}, metadata=metadata, created_at=now, last_modified=now
        )
        self.documents[doc_id] = doc
        self.storage.save_document(doc_id, doc)
        return doc

    def get_document(self, doc_id: str) -> Optional[DocumentContent]:
        """Retrieves a document by ID."""
        if doc := self.documents.get(doc_id):
            return doc
        raise DocumentNotFoundError(f"Document {doc_id} not found")

    def update_section(
        self,
        doc_id: str,
        section_id: str,
        title: str,
        content: str,
        context_tokens: int,
    ) -> DocumentSection:
        """Updates or creates a document section."""
        if not (doc := self.get_document(doc_id)):
            raise DocumentNotFoundError(f"Document {doc_id} not found")

        now = datetime.now()

        if section_id in doc.sections:
            section = doc.sections[section_id]
            section.title = title
            section.content = content
            section.last_modified = now
            section.version += 1
            section.context_tokens = context_tokens
        else:
            section = DocumentSection(
                title=title,
                content=content,
                last_modified=now,
                context_tokens=context_tokens,
            )
            doc.sections[section_id] = section

        doc.last_modified = now
        doc.version += 1

        self.storage.save_document(doc_id, doc)
        return section

    def organize_sections(self, doc_id: str, max_tokens: int = 2048) -> List[str]:
        """Organizes document sections to fit within token limits."""
        if not (doc := self.get_document(doc_id)):
            raise DocumentNotFoundError(f"Document {doc_id} not found")

        sections = list(doc.sections.items())
        organized: List[str] = []
        current_tokens = 0
        current_section = ""

        for section_id, section in sections:
            if current_tokens + section.context_tokens > max_tokens:
                if current_section:
                    organized.append(current_section)
                current_section = section_id
                current_tokens = section.context_tokens
            else:
                current_section = section_id
                current_tokens += section.context_tokens

        if current_section:
            organized.append(current_section)

        return organized

    def list_documents(self) -> Dict[str, Dict[str, str]]:
        """Lists all available documents with their metadata."""
        return self.storage.list_documents()
