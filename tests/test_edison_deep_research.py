import pytest
from edison.edison_deep_research import EdisonDeepResearch

@pytest.mark.unit
def test_edison_deep_research_initialization():
    """Test basic initialization of EdisonDeepResearch"""
    research = EdisonDeepResearch()
    assert isinstance(research, EdisonDeepResearch)