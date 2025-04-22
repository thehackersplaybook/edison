import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import set_default_openai_key, Agent, WebSearchTool, Runner
from .edison_agents import EdisonAgents, AgentType


DEFAULT_QNA_MODEL = "gpt-4o"
DEFAULT_LLM_MODEL = "gpt-4o"


class EdisonApiKeyConfig(BaseModel):
    """
    Configuration class for Edison API key.
    """

    openai_api_key: str
    firecrawl_api_key: str
    serper_api_key: str


class EdisonDeepResearch:
    """
    A simple, effective and powerful deep research module.
    """

    def __init__(self, api_key_config: EdisonApiKeyConfig = None):
        """
        Initialize the EdisonDeepResearch class.
        :param api_key_config: Configuration object containing API keys.
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
        self.init_agents()

    def init_agents(self):
        """
        Initialize the agents for deep research.
        """
        self.agents.tasks_agent = Agent(
            name="EdisonDeepResearch: Task Agent",
            instructions="""
                You are an AI agent that performs tasks based on the query provided to you.
                You will be provided with a query and you need to perform the task.
            """,
            model=DEFAULT_QNA_MODEL,
        )

        self.agents.qna_agent = Agent(
            name="EdisonDeepResearch: Questioning Agent",
            instructions="""
                You are an AI agent that asks more questions regarding a topic or query to get more information.
                You will be provided with a query and you need to ask more questions to get more information.
            """,
            model=DEFAULT_LLM_MODEL,
        )

        self.agents.summarizer_agent = Agent(
            name="EdisonDeepResearch: Summarizer Agent",
            instructions="""
                You are an AI agent that summarizes the information provided to you.
                You will be provided with a query and you need to summarize the information.
            """,
            model=DEFAULT_LLM_MODEL,
        )

        self.agents.generator_agent = Agent(
            name="EdisonDeepResearch: Generator Agent",
            instructions="""
                You are an AI agent that generates information based on the query provided to you.
                You will be provided with a query and you need to generate information.
            """,
            model=DEFAULT_LLM_MODEL,
            tools=[WebSearchTool()],
        )

        self.agents.expander_agent = Agent(
            name="EdisonDeepResearch: Query Expander Agent",
            instructions="""
                You are an AI agent that expands the query provided to you.
                You will be provided with a query and you need to expand it.
            """,
            model=DEFAULT_LLM_MODEL,
        )

        self.agents.set_agent(AgentType.TASKS_AGENT, self.agents.tasks_agent)
        self.agents.set_agent(AgentType.QNA_AGENT, self.agents.qna_agent)
        self.agents.set_agent(AgentType.SUMMARIZER_AGENT, self.agents.summarizer_agent)
        self.agents.set_agent(AgentType.GENERATOR_AGENT, self.agents.generator_agent)
        self.agents.set_agent(
            AgentType.QUERY_EXPANDER_AGENT, self.agents.expander_agent
        )
        print("Agents initialized successfully.")

    def are_agents_initialized(self):
        """
        Check if the agents are initialized.
        :return: True if agents are initialized, False otherwise.
        """
        return self.agents.are_agents_initialized()

    def get_agents(self) -> EdisonAgents:
        """
        Get the initialized agents.
        :return: The initialized agents.
        """
        return self.agents

    def _validate_api_keys(self):
        """
        Validate the API keys.
        :return: True if all API keys are valid, False otherwise.
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

    def _generate_questions(self, query: str):
        """
        Generate questions based on the given query.
        :param query: The query to generate questions for.
        :return: A list of generated questions.
        """
        pass

    def deep(self, query: str, model: str = DEFAULT_LLM_MODEL):
        """
        Perform a deep research on the given query using the specified model.
        :param query: The query to research.
        :param model: The model to use for research.
        :return: The result of the deep research.
        """
        pass
