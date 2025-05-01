"""QNA Engine module for Edison.

This module provides functionality to generate question-answer pairs
using AI agents for deep research and information gathering.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import traceback
from typing import List
from agents import Runner
from .models import QnaAgentOutput, QnaItem
from .edison_agents import EdisonAgents, AgentType


class QnaEngine:
    """A class to handle QNA generation operations using Edison agents.

    This class manages the generation of question-answer pairs
    using AI-powered agents for deep research.

    Attributes:
        agents (EdisonAgents): The Edison agents instance used for QNA generation.
    """

    def __init__(self, agents: EdisonAgents):
        """Initialize the QnaEngine.

        Args:
            agents (EdisonAgents): Instance of Edison agents to use for QNA generation.
        """
        self.agents = agents

    def generate_qna(self, queries: List[str]) -> List[QnaItem]:
        """Generate question-answer pairs for a given query.

        Args:
            queries (List[str]): List of queries to generate QNA pairs for.

        Returns:
            List[QnaItem]: List of QNA items containing question-answer pairs.
                       Each QNA item has 'question' and 'answer' attributes.

        Raises:
            Exception: If QNA generation fails, returns empty list.
        """
        try:
            qna_agent = self.agents.get_agent(AgentType.QNA_AGENT)
            formatted_queries = [f"{i+1}] {query}" for i, query in enumerate(queries)]
            result = Runner.run_sync(qna_agent, formatted_queries)
            qna_output: QnaAgentOutput = result.final_output
            return qna_output.qna_pairs
        except Exception as e:
            print(f"Error during QNA generation: {e}")
            traceback.print_exc()
            return []

    async def generate_qna_async(self, queries: List[str]) -> List[QnaItem]:
        """Asynchronously generate question-answer pairs for a given query.

        Args:
            queries (List[str]): List of queries to generate QNA pairs for.

        Returns:
            List[QnaItem]: List of QNA items containing question-answer pairs.
                       Each QNA item has 'question' and 'answer' attributes.

        Raises:
            Exception: If QNA generation fails, returns empty list.
        """
        try:
            qna_agent = self.agents.get_agent(AgentType.QNA_AGENT)
            formatted_queries = [f"{i+1}] {query}" for i, query in enumerate(queries)]
            result = await Runner.run(qna_agent, formatted_queries)
            qna_output: QnaAgentOutput = result.final_output
            return qna_output.qna_pairs
        except Exception as e:
            print(f"Error during QNA generation: {e}")
            traceback.print_exc()
            return []
