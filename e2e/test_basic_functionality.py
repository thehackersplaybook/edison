import pytest
from edison.edison_deep_research import EdisonDeepResearch, EdisonApiKeyConfig


@pytest.fixture(autouse=True)
def mock_env_keys(monkeypatch):
    """Fixture to set mock environment variables for all tests"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")


@pytest.mark.e2e
def test_end_to_end_basic_flow():
    """Test basic end-to-end functionality of EdisonDeepResearch"""
    # Initialize the research module using environment variables
    research = EdisonDeepResearch()

    # Verify API keys are loaded from environment
    assert research.api_key_config.openai_api_key == "test-openai-key"
    assert research.api_key_config.firecrawl_api_key == "test-firecrawl-key"
    assert research.api_key_config.serper_api_key == "test-serper-key"

    # Verify agents are properly initialized
    assert research.are_agents_initialized()
    agents = research.get_agents()
    assert agents.tasks_agent is not None
    assert agents.qna_agent is not None
    assert agents.summarizer_agent is not None
    assert agents.generator_agent is not None


@pytest.mark.e2e
def test_end_to_end_config_flow():
    """Test end-to-end functionality with explicit config"""
    # Initialize with explicit config
    config = EdisonApiKeyConfig(
        openai_api_key="test-openai-key",
        firecrawl_api_key="test-firecrawl-key",
        serper_api_key="test-serper-key",
    )
    research = EdisonDeepResearch(api_key_config=config)

    # Verify config is properly set
    assert research.api_key_config == config

    # Verify agents are initialized
    assert research.are_agents_initialized()
    agents = research.get_agents()
    assert agents.tasks_agent is not None
    assert agents.qna_agent is not None
    assert agents.summarizer_agent is not None
    assert agents.generator_agent is not None
