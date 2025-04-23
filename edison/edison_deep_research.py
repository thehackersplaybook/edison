"""Edison Deep Research module.

This module provides deep research capabilities by utilizing multiple AI agents
to analyze, question, summarize and generate information from queries.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import os
from dotenv import load_dotenv
from agents import set_default_openai_key
from .edison_agents import EdisonAgents
from .query_expander import QueryExpander
from .qna_engine import QnaEngine
from .models import EdisonApiKeyConfig


DEFAULT_QNA_MODEL = "gpt-4o"
DEFAULT_LLM_MODEL = "gpt-4o"


class EdisonDeepResearch:
    """A class to handle deep research operations using multiple specialized AI agents.

    This class manages multiple AI agents that work together to perform deep research
    on given queries through questioning, summarizing, and information generation.

    Attributes:
        api_key_config (EdisonApiKeyConfig): Configuration containing required API keys.
        agents (EdisonAgents): Collection of specialized AI agents used for research.
    """

    def __init__(self, api_key_config: EdisonApiKeyConfig = None):
        """Initialize the EdisonDeepResearch instance.

        Args:
            api_key_config (EdisonApiKeyConfig, optional): Configuration object containing API keys.
                If not provided, keys are loaded from environment variables.

        Raises:
            ValueError: If the provided API keys are invalid or missing.
        """
        if api_key_config:
            self.api_key_config = api_key_config
        else:
            load_dotenv(dotenv_path=".env", override=True)
            self.api_key_config = EdisonApiKeyConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
                serper_api_key=os.getenv("SERPER_API_KEY"),
            )

        if not self._validate_api_keys():
            raise ValueError(
                "Invalid API keys provided. Please check your configuration."
            )

        set_default_openai_key(self.api_key_config.openai_api_key)
        self.agents = EdisonAgents()
        self.agents.init_agents()

        self._query_expander = QueryExpander(agents=self.agents)
        self._qna_engine = QnaEngine(agents=self.agents)

    def are_agents_initialized(self):
        """Check if all required agents are properly initialized.

        Returns:
            bool: True if agents are initialized, False otherwise.
        """
        return self.agents.are_agents_initialized()

    def get_agents(self) -> EdisonAgents:
        """Get the collection of initialized agents.

        Returns:
            EdisonAgents: The initialized agents collection.
        """
        return self.agents

    def _validate_api_keys(self):
        """Validate the required API keys.

        Returns:
            bool: True if all API keys are valid, False otherwise.
        """
        if not self.api_key_config.openai_api_key:
            print("OpenAI API key is missing.")
            return False
        if not self.api_key_config.firecrawl_api_key:
            print("Firecrawl API key is missing.")
            return False
        if not self.api_key_config.serper_api_key:
            print("Serper API key is missing.")
            return False
        return True

    def deep(self, query: str, model: str = DEFAULT_LLM_MODEL):
        """Perform deep research on the given query.

        Args:
            query (str): The query to research.
            model (str, optional): The model to use for research. Defaults to DEFAULT_LLM_MODEL.

        Returns:
            Any: The result of the deep research.
        """
        print(f"Performing deep research on query: {query}")
        expanded_queries = self._query_expander.expand_query(query)
        print(f"Expanded queries: {expanded_queries}")
        qna_pairs = self._qna_engine.generate_qna(queries=expanded_queries)
        print(f"QnA pairs: {qna_pairs}")
        pass
