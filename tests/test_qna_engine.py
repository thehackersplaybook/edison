import pytest
from unittest.mock import Mock, patch
from edison.qna_engine import QnaEngine
from edison.edison_agents import EdisonAgents, AgentType
from edison.models import QnaAgentOutput, QnaItem
from agents import Runner


class MockAgent:
    """Mock Agent class for testing"""

    def __init__(self, name="mock_agent"):
        self.name = name


@pytest.fixture
def mock_edison_agents():
    """Fixture to create mocked EdisonAgents instance"""
    agents = EdisonAgents()
    mock_agent = MockAgent()
    agents.set_agent(AgentType.QNA_AGENT, mock_agent)
    return agents


@pytest.fixture
def qna_engine(mock_edison_agents):
    """Fixture to create QnaEngine instance with mocked agents"""
    return QnaEngine(agents=mock_edison_agents)


@pytest.mark.unit
def test_generate_qna_success(qna_engine):
    """Test successful QNA generation"""
    # Prepare test data
    test_queries = ["What is quantum computing?", "How does AI work?"]
    expected_qna = [
        QnaItem(
            question="What is quantum computing?", answer="A computing paradigm..."
        ),
        QnaItem(question="How does AI work?", answer="AI works by..."),
    ]
    mock_output = QnaAgentOutput(qna_pairs=expected_qna)

    # Mock Runner.run_sync
    with patch.object(Runner, "run_sync", return_value=Mock(final_output=mock_output)):
        result = qna_engine.generate_qna(test_queries)

        assert result == expected_qna
        assert len(result) == 2
        assert isinstance(result[0], QnaItem)


@pytest.mark.unit
def test_generate_qna_empty_queries(qna_engine):
    """Test QNA generation with empty queries list"""
    result = qna_engine.generate_qna([])
    assert result == []


@pytest.mark.unit
def test_generate_qna_error_handling(qna_engine):
    """Test error handling during QNA generation"""
    test_queries = ["Test query"]

    # Mock Runner.run_sync to raise an exception
    with patch.object(Runner, "run_sync", side_effect=Exception("Test error")):
        result = qna_engine.generate_qna(test_queries)
        assert result == []


@pytest.mark.asyncio
async def test_generate_qna_async_success(qna_engine):
    """Test successful async QNA generation"""
    test_queries = ["What is quantum computing?"]
    expected_qna = [
        QnaItem(question="What is quantum computing?", answer="A computing paradigm...")
    ]
    mock_output = QnaAgentOutput(qna_pairs=expected_qna)

    # Mock Runner.run
    with patch.object(Runner, "run", return_value=Mock(final_output=mock_output)):
        result = await qna_engine.generate_qna_async(test_queries)

        assert result == expected_qna
        assert len(result) == 1
        assert isinstance(result[0], QnaItem)


@pytest.mark.asyncio
async def test_generate_qna_async_error_handling(qna_engine):
    """Test error handling during async QNA generation"""
    test_queries = ["Test query"]

    # Mock Runner.run to raise an exception
    with patch.object(Runner, "run", side_effect=Exception("Test error")):
        result = await qna_engine.generate_qna_async(test_queries)
        assert result == []


@pytest.mark.unit
def test_query_formatting(qna_engine):
    """Test proper formatting of input queries"""
    test_queries = ["Query 1", "Query 2"]
    expected_formatted = ["1] Query 1", "2] Query 2"]
    mock_output = QnaAgentOutput(qna_pairs=[])

    with patch.object(
        Runner, "run_sync", return_value=Mock(final_output=mock_output)
    ) as mock_run:
        qna_engine.generate_qna(test_queries)
        # Verify the formatted queries were passed to the agent
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        assert args[1] == expected_formatted
