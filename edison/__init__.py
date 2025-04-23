"""Edison Research Module.

This module provides the main entry point for the Edison deep research system,
which utilizes advanced AI agents for comprehensive information analysis and generation.

The module exposes the EdisonDeepResearch class as its primary interface for
conducting in-depth research operations through multiple specialized AI agents.

Typical usage example:
    from edison import EdisonDeepResearch

    researcher = EdisonDeepResearch()
    result = researcher.deep("Impact of AI on healthcare")

Note:
    This module requires proper API configuration for OpenAI, Firecrawl, and Serper
    services before use. Keys can be provided via environment variables or direct
    configuration.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import warnings
from .edison_deep_research import EdisonDeepResearch

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main",
)

__version__ = "0.0.1"
__all__ = ["EdisonDeepResearch"]
