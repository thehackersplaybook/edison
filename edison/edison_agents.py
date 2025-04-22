from agents import Agent
from enum import Enum


class AgentType(Enum):
    """
    Enum to define the types of agents.
    """

    TASKS_AGENT = "tasks_agent"
    QNA_AGENT = "qna_agent"
    SUMMARIZER_AGENT = "summarizer_agent"
    GENERATOR_AGENT = "generator_agent"


class EdisonAgents:
    """
    Class to manage the agents for Edison.
    """

    def __init__(self):
        """
        Initialize the EdisonAgents class.
        """
        self.tasks_agent = None
        self.qna_agent = None
        self.summarizer_agent = None
        self.generator_agent = None

    def set_agent(self, agent_type: AgentType, agent: Agent) -> None:
        """
        Sets the agent.
        """
        if agent_type == AgentType.TASKS_AGENT:
            self.tasks_agent = agent
        elif agent_type == AgentType.QNA_AGENT:
            self.qna_agent = agent
        elif agent_type == AgentType.SUMMARIZER_AGENT:
            self.summarizer_agent = agent
        elif agent_type == AgentType.GENERATOR_AGENT:
            self.generator_agent = agent
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

    def get_agent(self, agent_type: AgentType) -> Agent:
        """
        Gets the agent.
        """
        if agent_type == AgentType.TASKS_AGENT:
            return self.tasks_agent
        elif agent_type == AgentType.QNA_AGENT:
            return self.qna_agent
        elif agent_type == AgentType.SUMMARIZER_AGENT:
            return self.summarizer_agent
        elif agent_type == AgentType.GENERATOR_AGENT:
            return self.generator_agent
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

    def are_agents_initialized(self) -> bool:
        """
        Check if the agents are initialized.
        :return: True if agents are initialized, False otherwise.
        """
        return (
            self.tasks_agent is not None
            and self.qna_agent is not None
            and self.summarizer_agent is not None
            and self.generator_agent is not None
        )
