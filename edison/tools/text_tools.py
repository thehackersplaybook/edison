"""Text Analysis Tools Module.

This module provides tools for document comparison, text analysis, and content processing.
It includes functionality for calculating text similarity using sequence matching and
merging document sections using AI assistance.

Typical usage example:
    analyzer = TextAnalyzer()
    similarity = analyzer.calculate_similarity("text1", "text2")
    merged = analyzer.merge_sections(section1, section2)

Note:
    The merge_sections method requires proper OpenAI API configuration.
    API key should be set via environment variable OPENAI_API_KEY.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import re
import os
from typing import Dict, Optional, Tuple
from difflib import SequenceMatcher
from openai import OpenAI
from ..models import DocumentSection, DocumentContent, ComparisonResult, MergeResult


class TextAnalyzer:
    """Tools for analyzing and comparing text content.

    This class provides methods for text similarity calculation and document section
    management. It uses sequence matching for basic text comparison and AI assistance
    for more complex merging operations.

    Attributes:
        openai: OpenAI client instance for AI-assisted operations
    """

    def __init__(self, openai_client: Optional[OpenAI] = None):
        """Initialize the text analyzer.

        Args:
            openai_client: Optional OpenAI client. If not provided, will attempt to create
                         one using environment variables.

        Raises:
            ValueError: If OpenAI client creation fails due to missing API key
        """
        self.openai = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using sequence matching.

        Args:
            text1: First text string to compare
            text2: Second text string to compare

        Returns:
            float: Similarity score between 0.0 and 1.0, where 1.0 indicates identical texts
        """
        if not text1 or not text2:
            return 0.0

        return SequenceMatcher(None, text1, text2).ratio()

    def find_most_relevant_section(
        self, doc: DocumentContent, title: str, content: str
    ) -> Tuple[Optional[str], float]:
        """Find most relevant existing section for given title/content.

        Args:
            doc: Document content containing sections to search through
            title: Title to compare against existing sections
            content: Content to compare against existing sections

        Returns:
            Tuple containing:
                - Optional[str]: ID of most relevant section, if found
                - float: Similarity score of the best match
        """
        if not doc.sections:
            return None, 0.0

        max_similarity = 0.0
        best_section_id = None

        for section_id, section in doc.sections.items():
            # Calculate similarity based on titles and content
            title_similarity = self.calculate_similarity(section.title, title)
            content_similarity = self.calculate_similarity(section.content, content)

            # Weight title similarity more heavily
            similarity = (title_similarity * 0.6) + (content_similarity * 0.4)

            if similarity > max_similarity:
                max_similarity = similarity
                best_section_id = section_id

        return best_section_id, max_similarity

    def compare_sections(
        self, section1: DocumentSection, section2: DocumentSection
    ) -> ComparisonResult:
        """Compare two document sections using sequence matching.

        Args:
            section1: First document section to compare
            section2: Second document section to compare

        Returns:
            ComparisonResult: Object containing similarity score and explanation
        """
        try:
            title_similarity = self.calculate_similarity(section1.title, section2.title)
            content_similarity = self.calculate_similarity(
                section1.content, section2.content
            )

            # Weight title and content similarities
            overall_similarity = (title_similarity * 0.4) + (content_similarity * 0.6)

            explanation = (
                f"Title similarity: {title_similarity:.2f}, "
                f"Content similarity: {content_similarity:.2f}"
            )

            return ComparisonResult(
                similarity_score=overall_similarity, explanation=explanation
            )
        except Exception as e:
            return ComparisonResult(
                similarity_score=0.0,
                explanation=f"Failed to compare sections: {str(e)}",
            )

    def merge_sections(
        self, section1: DocumentSection, section2: DocumentSection
    ) -> MergeResult:
        """Merge two document sections using AI assistance.

        This method uses OpenAI's API to intelligently merge the content and titles
        of two document sections while preserving the most important information
        from both.

        Args:
            section1: First document section to merge
            section2: Second document section to merge

        Returns:
            MergeResult: Object containing merged title, content, and source sections

        Raises:
            Exception: If AI-assisted merging fails
        """
        try:
            response = self.openai.responses.parse(
                model="gpt-4-turbo-preview",
                text_format=MergeResult,
                input=[
                    {
                        "role": "user",
                        "content": f"Merge these two document sections into a unified section with a structured response containing merged_title (string), merged_content (string), and source_sections (list of strings):\n\nSection 1 ({section1.title}):\n{section1.content}\n\nSection 2 ({section2.title}):\n{section2.content}",
                    }
                ],
            )
            return response.output_parsed
        except Exception as e:
            return MergeResult(
                merged_title=section1.title,
                merged_content=section1.content,
                source_sections=[section1.title],
            )
