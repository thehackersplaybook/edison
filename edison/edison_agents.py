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

from agents import Agent
from typing import Dict, Optional
from .edison_tools import EdisonTools
from .models import AgentType
from .agent_config import AGENT_CONFIGS


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
        self._tools = EdisonTools()

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

    @property
    def document_writer_agent(self) -> Optional[Agent]:
        return self._agents.get(AgentType.DOCUMENT_WRITER_AGENT)

    @property
    def orchestrator_agent(self) -> Optional[Agent]:
        return self._agents.get(AgentType.ORCHESTRATOR_AGENT)

    def init_agents(self) -> None:
        """Initializes all specialized AI agents for deep research operations.

        This method creates and configures each agent based on their respective
        configurations defined in AGENT_CONFIGS.

        Raises:
            ValueError: If agent initialization fails due to invalid configuration
        """
        for agent_type, config in AGENT_CONFIGS.items():
            tools = []
            handoffs = None
            agent = None

            if config.tools:
                tools = [self._tools.get_tool(tool_type) for tool_type in config.tools]

            if config.handoffs:
                handoffs = [self.get_agent(handoff) for handoff in config.handoffs]

            if handoffs:
                agent = Agent(
                    name=config.name,
                    instructions=config.instructions,
                    model=config.model,
                    tools=tools,
                    output_type=config.output_type,
                    handoffs=handoffs,
                )
            else:
                agent = Agent(
                    name=config.name,
                    instructions=config.instructions,
                    model=config.model,
                    tools=tools,
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
