"""Edison Agents Module.

This module provides a collection of specialized AI agents used for deep research operations.
Each agent is configured for specific tasks such as questioning, summarizing, and information
generation.

Typical usage example:
    agents = EdisonAgents()
    agents.init_agents()
    task_agent = agents.get_agent(AgentType.TASKS_AGENT)
    result = task_agent.run("Research quantum computing advances in 2023")

Note:
    All agents require proper API configuration before initialization.
    See EdisonDeepResearch class for API configuration details.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

from agents import Agent, WebSearchTool
from enum import Enum
from .models import ExpanderAgentOutput, QnaAgentOutput
from typing import Dict, Optional, List, Type, Any
from pydantic import BaseModel

DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_QNA_MODEL = "gpt-4o"


class AgentType(Enum):
    """Enumeration of available agent types in the Edison system.

    Each agent type represents a specialized AI agent with specific capabilities:
        TASKS_AGENT: Executes specific research tasks
        QNA_AGENT: Generates and answers questions
        SUMMARIZER_AGENT: Provides concise summaries
        GENERATOR_AGENT: Creates content and information
        QUERY_EXPANDER_AGENT: Expands search queries for broader coverage
    """

    TASKS_AGENT = "tasks_agent"
    QNA_AGENT = "qna_agent"
    SUMMARIZER_AGENT = "summarizer_agent"
    GENERATOR_AGENT = "generator_agent"
    QUERY_EXPANDER_AGENT = "query_expander_agent"


class AgentConfig(BaseModel):
    """Configuration model for Edison AI agents.

    Attributes:
        name (str): The display name of the agent
        instructions (str): The prompt/instructions for the agent's behavior
        model (str): The LLM model to be used by the agent
        tools (Optional[List[Any]]): List of tools available to the agent
        output_type (Optional[Type]): Expected output type for the agent's responses
    """

    name: str
    instructions: str
    model: str
    tools: Optional[List[Any]] = None
    output_type: Optional[Type] = None


AGENT_CONFIGS: Dict[AgentType, AgentConfig] = {
    AgentType.TASKS_AGENT: AgentConfig(
        name="EdisonDeepResearch: Task Agent",
        instructions="""
            You are an AI agent that performs tasks based on the query provided to you.
            You will be provided with a query and you need to perform the task.
        """,
        model=DEFAULT_QNA_MODEL,
    ),
    AgentType.QNA_AGENT: AgentConfig(
        name="EdisonDeepResearch: Questioning Agent",
        instructions="""
            You are an AI agent that asks more questions regarding a topic or query to get more information.
            You will be provided with a query and you need to ask more questions to get more information.
        """,
        model=DEFAULT_LLM_MODEL,
        output_type=QnaAgentOutput,
        tools=[
            WebSearchTool(),
        ],
    ),
    AgentType.SUMMARIZER_AGENT: AgentConfig(
        name="EdisonDeepResearch: Summarizer Agent",
        instructions="""
            You are an AI agent that summarizes the information provided to you.
            You will be provided with a query and you need to summarize the information.
        """,
        model=DEFAULT_LLM_MODEL,
    ),
    AgentType.GENERATOR_AGENT: AgentConfig(
        name="EdisonDeepResearch: Generator Agent",
        instructions="""
            You are an AI agent that generates information based on the query provided to you.
            You will be provided with a query and you need to generate information.
        """,
        model=DEFAULT_LLM_MODEL,
        tools=[WebSearchTool()],
    ),
    AgentType.QUERY_EXPANDER_AGENT: AgentConfig(
        name="EdisonDeepResearch: Query Expander Agent",
        instructions="""
            You are an AI agent that expands the query provided to you.
            You will be provided with a query and you need to expand it.
        """,
        model=DEFAULT_LLM_MODEL,
        output_type=ExpanderAgentOutput,
    ),
}


class EdisonAgents:
    """Manager class for Edison AI agents.

    This class handles the lifecycle and organization of various specialized AI agents
    used in deep research operations. It provides methods to initialize, access, and
    verify the status of these agents.

    Attributes:
        _agents (Dict[AgentType, Optional[Agent]]): Internal dictionary storing agent instances
    """

    def __init__(self):
        """Initializes an empty collection of Edison agents."""
        self._agents: Dict[AgentType, Optional[Agent]] = {
            agent_type: None for agent_type in AgentType
        }

    @property
    def tasks_agent(self) -> Optional[Agent]:
        return self._agents.get(AgentType.TASKS_AGENT)

    @property
    def qna_agent(self) -> Optional[Agent]:
        return self._agents.get(AgentType.QNA_AGENT)

    @property
    def summarizer_agent(self) -> Optional[Agent]:
        return self._agents.get(AgentType.SUMMARIZER_AGENT)

    @property
    def generator_agent(self) -> Optional[Agent]:
        return self._agents.get(AgentType.GENERATOR_AGENT)

    @property
    def query_expander_agent(self) -> Optional[Agent]:
        return self._agents.get(AgentType.QUERY_EXPANDER_AGENT)

    def init_agents(self) -> None:
        """Initializes all specialized AI agents for deep research operations.

        This method creates and configures each agent based on their respective
        configurations defined in AGENT_CONFIGS.

        Raises:
            ValueError: If agent initialization fails due to invalid configuration
        """
        for agent_type, config in AGENT_CONFIGS.items():
            agent = Agent(
                name=config.name,
                instructions=config.instructions,
                model=config.model,
                tools=config.tools,
                output_type=config.output_type,
            )
            self.set_agent(agent_type, agent)
        print("Agents initialized successfully.")

    def set_agent(self, agent_type: AgentType, agent: Agent) -> None:
        """Sets an agent instance for a specific agent type.

        Args:
            agent_type (AgentType): The type of agent to set
            agent (Agent): The agent instance to assign

        Raises:
            ValueError: If the agent_type is invalid
        """
        if not isinstance(agent_type, AgentType):
            raise ValueError(f"Invalid agent type: {agent_type}")
        self._agents[agent_type] = agent

    def get_agent(self, agent_type: AgentType) -> Agent:
        """Retrieves an agent instance by its type.

        Args:
            agent_type (AgentType): The type of agent to retrieve

        Returns:
            Agent: The requested agent instance

        Raises:
            ValueError: If the agent_type is invalid
        """
        if not isinstance(agent_type, AgentType):
            raise ValueError(f"Invalid agent type: {agent_type}")
        return self._agents[agent_type]

    def are_agents_initialized(self) -> bool:
        """Verifies if all agents are properly initialized.

        Returns:
            bool: True if all agents are initialized, False otherwise
        """
        return all(agent is not None for agent in self._agents.values())
