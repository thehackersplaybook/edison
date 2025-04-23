import pytest
from edison.edison_deep_research import EdisonDeepResearch, EdisonApiKeyConfig


@pytest.fixture(autouse=True)
def mock_env_keys(monkeypatch):
    """Fixture to set mock environment variables for all tests"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-firecrawl-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")


@pytest.fixture
def valid_api_config():
    """Fixture to create a valid API config"""
    return EdisonApiKeyConfig(
        openai_api_key="test-openai-key",
        firecrawl_api_key="test-firecrawl-key",
        serper_api_key="test-serper-key",
    )


@pytest.mark.unit
def test_edison_deep_research_initialization():
    """Test basic initialization of EdisonDeepResearch"""
    research = EdisonDeepResearch()
    assert isinstance(research, EdisonDeepResearch)


@pytest.mark.unit
@pytest.mark.parametrize(
    "openai_key,firecrawl_key,serper_key",
    [
        ("", "", ""),  # all empty
        ("valid-key", "", ""),  # only openai
        ("", "valid-key", ""),  # only firecrawl
        ("", "", "valid-key"),  # only serper
        ("valid-key", "valid-key", ""),  # missing serper
        ("valid-key", "", "valid-key"),  # missing firecrawl
        ("", "valid-key", "valid-key"),  # missing openai
    ],
)
def test_edison_initialization_with_invalid_config(
    openai_key, firecrawl_key, serper_key
):
    """Test initialization with various invalid API key configurations"""
    invalid_config = EdisonApiKeyConfig(
        openai_api_key=openai_key,
        firecrawl_api_key=firecrawl_key,
        serper_api_key=serper_key,
    )
    with pytest.raises(ValueError, match="Invalid API keys provided"):
        EdisonDeepResearch(api_key_config=invalid_config)


@pytest.mark.unit
def test_edison_initialization_with_env_vars():
    """Test initialization using environment variables"""
    research = EdisonDeepResearch()
    assert research.api_key_config.openai_api_key == "test-openai-key"
    assert research.api_key_config.firecrawl_api_key == "test-firecrawl-key"
    assert research.api_key_config.serper_api_key == "test-serper-key"


@pytest.mark.unit
def test_edison_initialization_with_config(valid_api_config):
    """Test initialization using config object"""
    research = EdisonDeepResearch(api_key_config=valid_api_config)
    assert research.api_key_config == valid_api_config


@pytest.mark.unit
def test_edison_invalid_api_keys():
    """Test initialization with invalid API keys"""
    invalid_config = EdisonApiKeyConfig(
        openai_api_key="", firecrawl_api_key="", serper_api_key=""
    )
    with pytest.raises(ValueError, match="Invalid API keys provided"):
        EdisonDeepResearch(api_key_config=invalid_config)


@pytest.mark.unit
def test_agents_initialization(valid_api_config):
    """Test that agents are properly initialized"""
    research = EdisonDeepResearch(api_key_config=valid_api_config)

    # Test initial state
    assert research.are_agents_initialized()

    # Test getting agents
    agents = research.get_agents()
    assert agents.tasks_agent is not None
    assert agents.qna_agent is not None
    assert agents.summarizer_agent is not None
    assert agents.generator_agent is not None
    assert agents.query_expander_agent is not None
