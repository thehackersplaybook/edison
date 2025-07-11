"""Document management tools with versioning and organization."""

import os
from typing import Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from ..models import DocumentContent, DocumentSection, ComparisonResult, MergeResult
from ..errors import DocumentNotFoundError
from .document_storage import DocumentStorage, ensure_dir
from .text_tools import TextAnalyzer


class DocumentWriterTool:
    """A tool for managing document content with versioning and organization."""

    def __init__(
        self, storage_dir: str = "documents", openai_client: Optional[OpenAI] = None
    ):
        """Initialize the document writer tool.

        Args:
            storage_dir: Path to directory for storing documents and their metadata
            openai_client: Optional OpenAI client for AI-assisted operations
        """
        ensure_dir(storage_dir)
        self.storage_dir = storage_dir
        self.storage = DocumentStorage(storage_dir)
        self.documents: Dict[str, DocumentContent] = {}
        self.text_analyzer = TextAnalyzer(openai_client=openai_client)
        self._load_existing_documents()

    def _load_existing_documents(self):
        """Load existing documents from storage."""
        for doc_id in self.storage.list_documents():
            try:
                self.documents[doc_id] = self.storage.load_document(doc_id)
            except Exception:
                continue

    def create_document(self, doc_id: str) -> DocumentContent:
        """Creates a new empty document and initializes it in storage.

        Args:
            doc_id: Unique identifier for the document

        Returns:
            The created document
        """
        now = datetime.now()
        doc = DocumentContent(
            sections={}, metadata=[], created_at=now, last_modified=now, version=0
        )
        self.documents[doc_id] = doc
        self.storage.save_document(doc_id, doc)
        return doc

    def get_document(self, doc_id: str) -> DocumentContent:
        """Retrieves a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            The requested document

        Raises:
            DocumentNotFoundError: If document does not exist
        """
        if doc_id not in self.documents:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        return self.documents[doc_id]

    def _compare_sections_with_ai(
        self, section1: DocumentSection, section2: DocumentSection
    ) -> Tuple[float, str]:
        """Compare two sections using AI.

        Args:
            section1: First section to compare
            section2: Second section to compare

        Returns:
            Tuple of (similarity score, explanation)
        """
        result = self.text_analyzer.compare_sections(section1, section2)
        return result.similarity_score, result.explanation

    def _merge_sections_with_ai(
        self, section1: DocumentSection, section2: DocumentSection
    ) -> Tuple[str, str]:
        """Merge two sections using AI.

        Args:
            section1: First section to merge
            section2: Second section to merge

        Returns:
            Tuple of (merged title, merged content)
        """
        result = self.text_analyzer.merge_sections(section1, section2)
        return result.merged_title, result.merged_content

    def update_section(self, doc_id: str, title: str, content: str) -> DocumentSection:
        """Updates or creates a document section.

        Args:
            doc_id: Document identifier
            title: Section title
            content: Section content in markdown format

        Returns:
            The updated or created section
        """

        doc = (
            self.get_document(doc_id)
            if doc_id in self.documents
            else self.create_document(doc_id)
        )
        now = datetime.now()

        # Try to find matching section
        matched_section_id, similarity = self.text_analyzer.find_most_relevant_section(
            doc, title, content
        )

        if matched_section_id and similarity > 0.7:
            # Found similar section, try to merge
            existing_section = doc.sections[matched_section_id]
            temp_section = DocumentSection(
                title=title, content=content, last_modified=now, version=0
            )

            # Try to merge the sections using AI
            merge_result = self.text_analyzer.merge_sections(
                existing_section, temp_section
            )
            section = DocumentSection(
                title=merge_result.merged_title,
                content=merge_result.merged_content,
                last_modified=now,
                version=existing_section.version + 1,
                context_tokens=len(merge_result.merged_content.split()),
            )
            doc.sections[matched_section_id] = section
        else:
            # Create new section
            section_id = f"section_{len(doc.sections) + 1}"
            section = DocumentSection(
                title=title,
                content=content,
                last_modified=now,
                version=0,
                context_tokens=len(content.split()),
            )
            doc.sections[section_id] = section

        # Update document metadata
        doc.last_modified = now
        doc.version += 1

        # Save document and markdown
        self.storage.save_document(doc_id, doc)
        self._write_markdown(doc_id, doc)

        return section

    def _write_markdown(self, doc_id: str, doc: DocumentContent):
        """Write document content to markdown file.

        Args:
            doc_id: Document identifier
            doc: Document to write
        """
        markdown_path = Path(self.storage_dir) / f"{doc_id}.md"

        content = []
        for section in doc.sections.values():
            # Add section title as h2
            content.append(f"\n## {section.title}\n")
            # Add section content, preserving existing markdown
            content.append(f"{section.content}\n")

        markdown_path.write_text("\n".join(content))
