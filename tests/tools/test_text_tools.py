"""Tests for text analysis tools."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from edison.tools.text_tools import TextAnalyzer
from edison.models import (
    DocumentContent,
    DocumentSection,
    ComparisonResult,
    MergeResult,
)


@pytest.fixture
def mock_openai():
    """Provides a mocked OpenAI client."""
    with patch("openai.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client

        # Mock the responses.parse method
        parse_response = MagicMock()
        parse_response.output_parsed = 0.9  # Default similarity score
        client.responses.parse = MagicMock(return_value=parse_response)

        yield client


@pytest.fixture
def analyzer(mock_openai):
    """Provides a TextAnalyzer instance with mocked OpenAI."""
    return TextAnalyzer(openai_client=mock_openai)


def test_calculate_similarity(analyzer, mock_openai):
    """Test text similarity calculation."""

    # Setup parse response for different scenarios
    def mock_parse(**kwargs):
        text1 = (
            kwargs["input"][0]["content"]
            .split("Text 1:")[1]
            .split("Text 2:")[0]
            .strip()
        )
        text2 = kwargs["input"][0]["content"].split("Text 2:")[1].strip()

        response = MagicMock()
        if text1 == text2:
            response.output_parsed = 1.0
        elif not text1 or not text2:
            response.output_parsed = 0.0
        else:
            response.output_parsed = 0.5
        return response

    mock_openai.responses.parse.side_effect = mock_parse

    # Test cases
    assert analyzer.calculate_similarity("hello world", "hello world") == 1.0
    assert analyzer.calculate_similarity("Hello World", "hello world") > 0.4
    assert 0.3 < analyzer.calculate_similarity("hello universe", "hello world") < 0.8
    assert analyzer.calculate_similarity("hello world", "goodbye moon") < 0.6
    assert analyzer.calculate_similarity("", "") == 0.0
    assert analyzer.calculate_similarity("hello", "") == 0.0


def test_find_most_relevant_section(analyzer):
    """Test finding relevant sections in documents."""
    now = datetime.now()

    # Create test document with sections
    doc = DocumentContent(
        sections={
            "section_1": DocumentSection(
                title="Introduction",
                content="This is an introduction to the topic.",
                last_modified=now,
            ),
            "section_2": DocumentSection(
                title="Methods",
                content="These are the methods we used.",
                last_modified=now,
            ),
        },
        metadata=[],
        created_at=now,
        last_modified=now,
    )

    # Test exact match
    section_id, score = analyzer.find_most_relevant_section(
        doc, "Introduction", "This is an introduction to the topic."
    )
    assert section_id == "section_1"
    assert score > 0.9

    # Test similar content
    section_id, score = analyzer.find_most_relevant_section(
        doc, "Intro", "This is an intro to the topic."
    )
    assert section_id == "section_1"
    assert score > 0.4

    # Test no match
    section_id, score = analyzer.find_most_relevant_section(
        doc, "Conclusion", "This is a completely different section."
    )
    assert score < 0.55

    # Test empty document
    empty_doc = DocumentContent(
        sections={}, metadata=[], created_at=now, last_modified=now
    )
    section_id, score = analyzer.find_most_relevant_section(
        empty_doc, "Title", "Content"
    )
    assert section_id is None
    assert score == 0.0


def test_compare_sections_with_structured_output(analyzer, mock_openai):
    """Test section comparison using structured outputs."""
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

    # Mock responses.parse return value
    mock_result = ComparisonResult(
        similarity_score=0.9,
        explanation="The sections are nearly identical with minor wording differences.",
    )
    parse_response = MagicMock()
    parse_response.output_parsed = mock_result
    mock_openai.responses.parse.return_value = parse_response

    result = analyzer.compare_sections(section1, section2)
    assert isinstance(result, ComparisonResult)
    assert result.similarity_score == 0.9
    assert "nearly identical" in result.explanation

    # Verify OpenAI was called with correct parameters
    mock_openai.responses.parse.assert_called_once()
    call_args = mock_openai.responses.parse.call_args[1]
    assert call_args["model"] == "gpt-4-turbo-preview"
    assert call_args["text_format"] == ComparisonResult
    assert "Compare these two document sections" in call_args["input"][0]["content"]


def test_merge_sections_with_structured_output(analyzer, mock_openai):
    """Test section merging using structured outputs."""
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

    # Mock responses.parse return value
    mock_result = MergeResult(
        merged_title="Research Methodology",
        merged_content="1. First step (Initial phase)\n2. Second step (Second phase)",
        source_sections=["Methods", "Methodology"],
    )
    parse_response = MagicMock()
    parse_response.output_parsed = mock_result
    mock_openai.responses.parse.return_value = parse_response

    result = analyzer.merge_sections(section1, section2)
    assert isinstance(result, MergeResult)
    assert result.merged_title == "Research Methodology"
    assert "First step" in result.merged_content
    assert "Second phase" in result.merged_content
    assert len(result.source_sections) == 2

    # Verify OpenAI was called with correct parameters
    call_args = mock_openai.responses.parse.call_args[1]
    assert call_args["model"] == "gpt-4-turbo-preview"
    assert call_args["text_format"] == MergeResult
    assert "Merge these two document sections" in call_args["input"][0]["content"]


def test_invalid_openai_responses(analyzer, mock_openai):
    """Test handling of invalid OpenAI responses."""
    section = DocumentSection(
        title="Test",
        content="Content",
        last_modified=datetime.now(),
    )

    # Test API error
    mock_openai.responses.parse.side_effect = Exception("API Error")

    # calculate_similarity doesn't use OpenAI, so it should work normally
    similarity = analyzer.calculate_similarity("text1", "text2")
    assert similarity > 0.0

    # Should return default ComparisonResult for compare_sections
    result = analyzer.compare_sections(section, section)
    assert isinstance(result, ComparisonResult)
    assert result.similarity_score == 0.0
    assert "Failed to compare sections" in result.explanation

    # Should return default MergeResult for merge_sections
    result = analyzer.merge_sections(section, section)
    assert isinstance(result, MergeResult)
    assert result.merged_title == section.title
    assert result.merged_content == section.content
    assert len(result.source_sections) == 1


def test_structured_output_validation(analyzer, mock_openai):
    """Test validation of structured outputs."""
    section = DocumentSection(
        title="Test", content="Content", last_modified=datetime.now()
    )

    # Test invalid similarity score type
    mock_openai.responses.parse.side_effect = ValueError("Invalid format")
    result = analyzer.compare_sections(section, section)
    assert result.similarity_score == 0.0
    assert "Failed to compare sections" in result.explanation

    # Test missing required field
    mock_openai.responses.parse.side_effect = ValueError("Missing required field")
    result = analyzer.merge_sections(section, section)
    assert result.merged_title == section.title
    assert result.merged_content == section.content
