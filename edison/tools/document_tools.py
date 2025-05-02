"""Document management tools for Edison agents.

This module provides tools for managing documents, including versioning,
content organization, and automatic restructuring.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from difflib import SequenceMatcher
import re
from ..errors import DocumentNotFoundError
from ..models import DocumentContent, DocumentSection, DocumentMetdataItem
from .document_storage import DocumentStorage
from ..common.utils import ensure_dir
from openai import OpenAI


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
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _load_existing_documents(self) -> None:
        """Load any existing documents from storage."""
        for doc_id in self.storage.list_documents():
            if doc := self.storage.load_document(doc_id):
                self.documents[doc_id] = doc

    def create_document(
        self, doc_id: str, metadata: List[DocumentMetdataItem]
    ) -> DocumentContent:
        """Creates a new empty document."""
        now = datetime.now()
        doc = DocumentContent(sections={}, created_at=now, last_modified=now)
        self.documents[doc_id] = doc
        self.storage.save_document(doc_id, doc)
        return doc

    def get_document(self, doc_id: str) -> Optional[DocumentContent]:
        """Retrieves a document by ID."""
        if doc := self.documents.get(doc_id):
            return doc
        raise DocumentNotFoundError(f"Document {doc_id} not found")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two text strings.

        Uses a combination of simple ratio and token-based similarity.
        """
        # Clean and normalize texts
        text1 = re.sub(r"\s+", " ", text1.lower().strip())
        text2 = re.sub(r"\s+", " ", text2.lower().strip())

        # Get base similarity
        similarity = SequenceMatcher(None, text1, text2).ratio()

        # Get token-based similarity
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        token_similarity = len(tokens1.intersection(tokens2)) / max(
            len(tokens1), len(tokens2)
        )

        # Return weighted average
        return (similarity * 0.6) + (token_similarity * 0.4)

    def _find_most_relevant_section(
        self, doc: DocumentContent, title: str, content: str
    ) -> Tuple[Optional[str], float]:
        """Find the most relevant existing section for the given content.

        Returns:
            Tuple of (section_id, similarity_score)
        """
        best_match = None
        best_score = 0.0

        for section_id, section in doc.sections.items():
            # Calculate similarity scores
            title_similarity = self._calculate_similarity(section.title, title)
            content_similarity = self._calculate_similarity(section.content, content)

            # Weight title more heavily than content
            combined_score = (title_similarity * 0.4) + (content_similarity * 0.6)

            if (
                combined_score > best_score and combined_score > 0.7
            ):  # Threshold for similarity
                best_score = combined_score
                best_match = section_id

        return best_match, best_score

    def update_section(
        self,
        doc_id: str,
        title: str,
        content: str,
    ) -> DocumentSection:
        """Updates or creates a document section.

        If section_id is None, attempts to find the most relevant existing section.
        """
        if not (doc := self.get_document(doc_id)):
            doc = self.create_document(doc_id, metadata=[])

        now = datetime.now()

        # If no section_id provided, try to find the most relevant section

        matched_section_id, _ = self._find_most_relevant_section(doc, title, content)
        section_id = matched_section_id or f"section_{len(doc.sections) + 1}"
        context_tokens = len(content.split())

        new_title = title
        new_content = content

        try:
            existing_section = doc.sections.get(section_id)
            response = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are given 2 sections, combine them into one section. ",
                    },
                    {
                        "role": "user",
                        "content": f"""
                        New Section Title: {title}
                        New Section Content: {content}

                        Existing Section Title: {existing_section.title if existing_section else 'N/A'}
                        Existing Section Content: {existing_section.content if existing_section else 'N/A'}
                        
                        Please combine the new section with the existing section, if it exists.
                        If the existing section is not relevant, create a new section with the new content.
                        Provide only the new section content without any additional information.
                        """,
                    },
                ],
            )

            new_content = response.choices[0].message.content
            new_title = title if not existing_section else existing_section.title
        except Exception as e:
            print(f"LLM error when combining sections: {e}")
            new_content = content
            new_title = title

        if section_id in doc.sections:
            section = doc.sections[section_id]
            section.title = new_title
            section.content = new_content
            section.last_modified = now
            section.version += 1
            section.context_tokens = context_tokens
        else:
            section = DocumentSection(
                title=new_title,
                content=new_content,
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
