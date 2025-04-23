import pytest
from unittest.mock import Mock, patch
from agents import Runner
from edison.query_expander import QueryExpander
from edison.edison_agents import EdisonAgents
from edison.models import ExpanderAgentOutput


class MockResult:
    def __init__(self, related_queries):
        self.final_output = ExpanderAgentOutput(related_queries=related_queries)


@pytest.fixture
def mock_agents():
    """Fixture to create mocked EdisonAgents instance"""
    agents = EdisonAgents()
    agents.expander_agent = Mock()
    return agents


@pytest.fixture
def query_expander(mock_agents):
    """Fixture to create QueryExpander instance with mocked agents"""
    return QueryExpander(mock_agents)


@pytest.mark.unit
def test_expand_query_success(query_expander):
    """Test successful synchronous query expansion"""
    expected_queries = ["query1", "query2", "query3"]
    with patch.object(Runner, "run_sync", return_value=MockResult(expected_queries)):
        result = query_expander.expand_query("test query")
        assert result == expected_queries


@pytest.mark.unit
def test_expand_query_failure(query_expander):
    """Test query expansion when an error occurs"""
    with patch.object(Runner, "run_sync", side_effect=Exception("Test error")):
        result = query_expander.expand_query("test query")
        assert result == "test query"  # Should return original query on error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expand_query_async_success(query_expander):
    """Test successful asynchronous query expansion"""
    expected_queries = ["query1", "query2", "query3"]
    with patch.object(Runner, "run", return_value=MockResult(expected_queries)):
        result = await query_expander.expand_query_async("test query")
        assert result == expected_queries


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expand_query_async_failure(query_expander):
    """Test asynchronous query expansion when an error occurs"""
    with patch.object(Runner, "run", side_effect=Exception("Test error")):
        result = await query_expander.expand_query_async("test query")
        assert result == "test query"  # Should return original query on error
