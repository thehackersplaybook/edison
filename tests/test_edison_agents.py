import pytest
from agents import Agent
from edison.edison_agents import EdisonAgents, AgentType


class MockAgent(Agent):
    """Mock Agent class for testing"""

    def __init__(self, name="mock_agent"):
        self.name = name
        self.handoffs = []

    def __eq__(self, other):
        if not isinstance(other, MockAgent):
            return False
        return self.name == other.name


@pytest.fixture
def edison_agents():
    """Fixture to create EdisonAgents instance for tests"""
    return EdisonAgents()


@pytest.mark.unit
def test_initial_state(edison_agents):
    """Test initial state of EdisonAgents"""
    assert edison_agents.tasks_agent is None
    assert edison_agents.qna_agent is None
    assert edison_agents.summarizer_agent is None
    assert edison_agents.generator_agent is None
    assert not edison_agents.are_agents_initialized()


@pytest.mark.unit
def test_set_and_get_agent(edison_agents):
    """Test setting and getting agents"""
    mock_agent = MockAgent()

    # Test setting each type of agent
    edison_agents.set_agent(AgentType.TASKS_AGENT, mock_agent)
    assert edison_agents.get_agent(AgentType.TASKS_AGENT) == mock_agent

    edison_agents.set_agent(AgentType.QNA_AGENT, mock_agent)
    assert edison_agents.get_agent(AgentType.QNA_AGENT) == mock_agent

    edison_agents.set_agent(AgentType.SUMMARIZER_AGENT, mock_agent)
    assert edison_agents.get_agent(AgentType.SUMMARIZER_AGENT) == mock_agent

    edison_agents.set_agent(AgentType.GENERATOR_AGENT, mock_agent)
    assert edison_agents.get_agent(AgentType.GENERATOR_AGENT) == mock_agent


@pytest.mark.unit
def test_invalid_agent_type(edison_agents):
    """Test handling of invalid agent types"""
    mock_agent = MockAgent()
    with pytest.raises(ValueError, match="Invalid agent type:.*"):
        edison_agents.set_agent("invalid_type", mock_agent)

    with pytest.raises(ValueError, match="Invalid agent type:.*"):
        edison_agents.get_agent("invalid_type")


@pytest.mark.unit
def test_are_agents_initialized_partial(edison_agents):
    """Test are_agents_initialized with partially initialized agents"""
    mock_agent = MockAgent()

    # Set only some agents
    edison_agents.set_agent(AgentType.TASKS_AGENT, mock_agent)
    assert not edison_agents.are_agents_initialized()

    edison_agents.set_agent(AgentType.QNA_AGENT, mock_agent)
    assert not edison_agents.are_agents_initialized()

    edison_agents.set_agent(AgentType.SUMMARIZER_AGENT, mock_agent)
    assert not edison_agents.are_agents_initialized()


@pytest.mark.unit
def test_are_agents_initialized_complete(edison_agents):
    """Test are_agents_initialized with all agents initialized"""
    mock_agent = MockAgent()

    # Set all agents
    edison_agents.set_agent(AgentType.TASKS_AGENT, mock_agent)
    edison_agents.set_agent(AgentType.QNA_AGENT, mock_agent)
    edison_agents.set_agent(AgentType.SUMMARIZER_AGENT, mock_agent)
    edison_agents.set_agent(AgentType.GENERATOR_AGENT, mock_agent)
    edison_agents.set_agent(AgentType.QUERY_EXPANDER_AGENT, mock_agent)

    assert edison_agents.are_agents_initialized()


@pytest.mark.unit
def test_get_uninitialized_agent(edison_agents):
    """Test getting an uninitialized agent"""
    assert edison_agents.get_agent(AgentType.TASKS_AGENT) is None
    assert edison_agents.get_agent(AgentType.QNA_AGENT) is None
    assert edison_agents.get_agent(AgentType.SUMMARIZER_AGENT) is None
    assert edison_agents.get_agent(AgentType.GENERATOR_AGENT) is None
