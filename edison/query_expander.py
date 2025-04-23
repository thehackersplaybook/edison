"""Query expansion module for Edison.

This module provides functionality to expand search queries using AI agents
to generate related and contextually relevant search terms.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import traceback
from typing import List
from agents import Runner
from .models import ExpanderAgentOutput
from .edison_agents import EdisonAgents


class QueryExpander:
    """A class to handle query expansion operations using Edison agents.

    This class manages the expansion of search queries into related terms
    using AI-powered agents.

    Attributes:
        agents (EdisonAgents): The Edison agents instance used for query expansion.
    """

    def __init__(self, agents: EdisonAgents):
        """Initialize the QueryExpander.

        Args:
            agents (EdisonAgents): Instance of Edison agents to use for expansion.
        """
        self.agents = agents

    def expand_query(self, query: str) -> List[str]:
        """Expand a query into related search terms.

        Args:
            query (str): The original query to expand.

        Returns:
            str: The expanded query text.

        Raises:
            Exception: If query expansion fails, returns original query.
        """
        try:
            result = Runner.run_sync(
                self.agents.expander_agent,
            )
            expander_agent_output: ExpanderAgentOutput = result.final_output
            return expander_agent_output.related_queries
        except Exception as e:
            print(f"Error during query expansion: {e}")
            traceback.print_exc()
            return query

    async def expand_query_async(self, query: str) -> List[str]:
        """Asynchronously expand a query into related search terms.

        Args:
            query (str): The original query to expand.

        Returns:
            str: The expanded query text.

        Raises:
            Exception: If query expansion fails, returns original query.
        """
        try:
            result = await Runner.run(
                self.agents.expander_agent,
            )
            expander_agent_output: ExpanderAgentOutput = result.final_output
            return expander_agent_output.related_queries
        except Exception as e:
            print(f"Error during query expansion: {e}")
            traceback.print_exc()
            return query
