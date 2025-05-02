"""Tests for document management tools."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from edison.tools.document_tools import DocumentWriterTool
from edison.models import (
    DocumentContent,
    DocumentSection,
    ComparisonResult,
    MergeResult,
    DocumentMetdataItem,
)
from edison.errors import DocumentNotFoundError


@pytest.fixture
def mock_openai():
    """Provides a mocked OpenAI client."""
    with patch("openai.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client

        # Mock the responses.parse method
        parse_response = MagicMock()
        parse_response.output_parsed = MergeResult(
            merged_title="Test Section",
            merged_content="This is test content",
            source_sections=["section_1"],
        )
        client.responses.parse = MagicMock(return_value=parse_response)

        yield client


@pytest.fixture
def storage_dir(tmp_path):
    """Provides a temporary directory for storage tests."""
    return str(tmp_path / "test_documents")


@pytest.fixture
def document_tool(storage_dir, mock_openai):
    """Provides a DocumentWriterTool instance with mocked OpenAI."""
    return DocumentWriterTool(storage_dir=storage_dir, openai_client=mock_openai)


def test_create_document(document_tool):
    """Test document creation."""
    doc = document_tool.create_document("test_doc")
    assert isinstance(doc, DocumentContent)
    assert len(doc.sections) == 0
    assert doc.version == 0


def test_get_document(document_tool):
    """Test document retrieval."""
    # Create a document first
    doc_id = "test_doc"
    document_tool.create_document(doc_id)

    # Test successful retrieval
    doc = document_tool.get_document(doc_id)
    assert isinstance(doc, DocumentContent)

    # Test non-existent document
    with pytest.raises(DocumentNotFoundError):
        document_tool.get_document("nonexistent")


def test_update_section(document_tool, mock_openai):
    """Test section updates."""
    doc_id = "test_doc"
    title = "Test Section"
    content = "This is test content"

    # Configure mock response specifically for this test
    mock_openai.responses.parse.return_value.output_parsed = MergeResult(
        merged_title=title, merged_content=content, source_sections=["section_1"]
    )

    # Test creating new section
    section = document_tool.update_section(doc_id, title, content)
    assert section.title == title
    assert section.content == content
    assert section.version == 0

    # Configure mock response for update
    new_content = "This is a text content with updates"
    mock_openai.responses.parse.return_value.output_parsed = MergeResult(
        merged_title=title, merged_content=new_content, source_sections=["section_1"]
    )

    # Test updating existing section
    updated = document_tool.update_section(doc_id, title, new_content)
    assert updated.content == new_content
    assert updated.version == 1


def test_markdown_generation(document_tool):
    """Test markdown file generation."""
    doc_id = "test_doc"

    # Add multiple sections
    document_tool.update_section(
        doc_id, "Introduction", "# Welcome\nThis is the intro."
    )
    document_tool.update_section(
        doc_id, "Methods", "## Methods\n1. First step\n2. Second step"
    )

    # Check markdown file creation
    markdown_path = Path(document_tool.storage_dir) / f"{doc_id}.md"
    assert markdown_path.exists()

    # Verify content
    content = markdown_path.read_text()
    assert "## Introduction" in content
    assert "## Methods" in content
    assert "# Welcome" in content
    assert "1. First step" in content


@pytest.mark.parametrize(
    "doc_id,title,content",
    [
        ("", "Title", "Content"),  # Empty doc_id
        ("doc", "", "Content"),  # Empty title
        ("doc", "Title", ""),  # Empty content
    ],
)
def test_edge_cases(document_tool, doc_id, title, content):
    """Test edge cases with empty or invalid inputs."""
    section = document_tool.update_section(doc_id, title, content)
    assert section is not None
    assert section.title == title
    assert section.content == content


def test_concurrent_updates(document_tool, mock_openai):
    """Test handling multiple updates to same section."""
    doc_id = "test_doc"
    title = "Test Section"

    # Configure mock response for initial section
    mock_openai.responses.parse.return_value.output_parsed = MergeResult(
        merged_title=title,
        merged_content="Initial content",
        source_sections=["section_1"],
    )

    # Create initial section
    document_tool.update_section(doc_id, title, "Initial content")

    # Configure mock for first update
    mock_openai.responses.parse.return_value.output_parsed = MergeResult(
        merged_title=title, merged_content="Update 1", source_sections=["section_1"]
    )
    section1 = document_tool.update_section(doc_id, title, "Update 1")

    # Configure mock for second update
    mock_openai.responses.parse.return_value.output_parsed = MergeResult(
        merged_title=title, merged_content="Update 2", source_sections=["section_1"]
    )
    section2 = document_tool.update_section(doc_id, title, "Update 2")

    assert section2.version > section1.version
    assert section2.content == "Update 2"


def test_special_characters(document_tool):
    """Test handling of special characters in content."""
    doc_id = "test_doc"
    title = "Special chars: !@#$%^&*()"
    content = """# Special content
    * List item with é and ñ
    * Unicode: 🚀 💻 🔥
    """

    section = document_tool.update_section(doc_id, title, content)
    assert section.title == title
    assert section.content == content

    # Verify markdown file handles special chars
    markdown_path = Path(document_tool.storage_dir) / f"{doc_id}.md"
    assert markdown_path.exists()
    saved_content = markdown_path.read_text()
    assert "🚀" in saved_content
    assert "é" in saved_content


def test_section_comparison_with_ai(document_tool, mock_openai):
    """Test section comparison using OpenAI."""
    section1 = DocumentSection(
        title="Introduction",
        content="This is an introduction.",
        last_modified=datetime.now(),
    )
    section2 = DocumentSection(
        title="Intro",
        content="This is an intro.",
        last_modified=datetime.now(),
    )

    # Configure mock response for comparison
    mock_result = ComparisonResult(
        similarity_score=0.9,
        explanation="The sections contain very similar content with minor wording differences.",
    )
    parse_response = MagicMock()
    parse_response.output_parsed = mock_result
    mock_openai.responses.parse.return_value = parse_response

    score, explanation = document_tool._compare_sections_with_ai(section1, section2)
    assert score == 0.9
    assert "very similar content" in explanation

    # Verify OpenAI was called with correct parameters
    call_args = mock_openai.responses.parse.call_args[1]
    assert call_args["text_format"] == ComparisonResult
    assert "Compare these two document sections" in call_args["input"][0]["content"]
    assert section1.title in call_args["input"][0]["content"]
    assert section2.title in call_args["input"][0]["content"]


def test_section_merging_with_ai(document_tool, mock_openai):
    """Test section merging using OpenAI."""
    section1 = DocumentSection(
        title="Methods",
        content="1. First step\n2. Second step",
        last_modified=datetime.now(),
    )
    section2 = DocumentSection(
        title="Methodology",
        content="1. Initial phase\n2. Second phase",
        last_modified=datetime.now(),
    )

    # Configure mock response for merging
    mock_result = MergeResult(
        merged_title="Research Methodology",
        merged_content="1. First step (Initial phase)\n2. Second step (Second phase)",
        source_sections=["Methods", "Methodology"],
    )
    parse_response = MagicMock()
    parse_response.output_parsed = mock_result
    mock_openai.responses.parse.return_value = parse_response

    merged_title, merged_content = document_tool._merge_sections_with_ai(
        section1, section2
    )
    assert merged_title == "Research Methodology"
    assert "First step" in merged_content
    assert "Second phase" in merged_content

    # Verify OpenAI was called with correct parameters
    call_args = mock_openai.responses.parse.call_args[1]
    assert call_args["text_format"] == MergeResult
    assert "Merge these two document sections" in call_args["input"][0]["content"]
    assert section1.title in call_args["input"][0]["content"]
    assert section2.title in call_args["input"][0]["content"]


def test_invalid_ai_response_handling(document_tool, mock_openai):
    """Test handling of invalid AI responses."""
    section1 = DocumentSection(
        title="Test",
        content="Content 1",
        last_modified=datetime.now(),
    )
    section2 = DocumentSection(
        title="Test",
        content="Content 2",
        last_modified=datetime.now(),
    )

    # Test API error
    mock_openai.responses.parse.side_effect = Exception("API Error")

    # Should handle error gracefully for comparison
    score, explanation = document_tool._compare_sections_with_ai(section1, section2)
    assert score == 0.0
    assert "Failed to compare sections" in explanation

    # Should handle error gracefully for merging
    title, content = document_tool._merge_sections_with_ai(section1, section2)
    assert title == section1.title
    assert content == section1.content
