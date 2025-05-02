"""Edison Deep Research module.

This module provides deep research capabilities by utilizing multiple AI agents
to analyze, question, summarize and generate information from queries.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

import os
import traceback
from typing import AsyncGenerator
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents.result import RunResultStreaming
from agents import set_default_openai_key, Runner, ItemHelpers
from .edison_agents import EdisonAgents
from .models import EdisonApiKeyConfig, AgentType
from .common.utils import generate_document_id


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

    def __init__(
        self, api_key_config: EdisonApiKeyConfig = None, dotenv_path: str = ".env"
    ):
        """Initialize the EdisonDeepResearch instance.

        Args:
            api_key_config (EdisonApiKeyConfig, optional): Configuration object containing API keys.
                If not provided, keys are loaded from environment variables.
            dotenv_path (str, optional): Path to the .env file. Defaults to ".env".

        Raises:
            ValueError: If the provided API keys are invalid or missing.
        """
        if api_key_config:
            self.api_key_config = api_key_config
            os.environ["OPENAI_API_KEY"] = api_key_config.openai_api_key
            os.environ["FIRECRAWL_API_KEY"] = api_key_config.firecrawl_api_key
            os.environ["SERPER_API_KEY"] = api_key_config.serper_api_key
        else:
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path=dotenv_path, override=True)
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

    async def deep_stream_async(
        self, query: str, model: str = DEFAULT_LLM_MODEL
    ) -> AsyncGenerator[str, None]:
        """Perform deep research on the given query and return a stream of results.

        Args:
            query (str): The query to research.
            model (str, optional): The model to use for research. Defaults to DEFAULT_LLM_MODEL.

        Yields:
            str: A stream of detailed agent run information and research results.
        """
        try:
            document_id = generate_document_id()
            orchestrator_agent = self.agents.get_agent(
                agent_type=AgentType.ORCHESTRATOR_AGENT
            )
            result: RunResultStreaming = Runner.run_streamed(
                orchestrator_agent,
                input=f"Deep research on: '{query}'. Document ID: '{document_id}'",
            )

            yield f"=== Deep Research Starting: Document ID: {document_id} ===\n"

            async for event in result.stream_events():
                if event.type == "agent_updated_stream_event":
                    yield f"Agent updated: {event.new_agent.name}\n"
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        yield f"-- Tool was called: [{event.item.raw_item.name}] | Input: {event.item.raw_item.arguments} | Call ID: {event.item.raw_item.call_id} |\n"
                    elif event.item.type == "tool_call_output_item":
                        yield f"-- Tool output: {event.item.output}\n"
                    elif event.item.type == "message_output_item":
                        yield f"-- Message output:\n{ItemHelpers.text_message_output(event.item)}\n"
                elif event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    yield event.data.delta

            yield f"\n=== Deep Research Complete: Document ID = {document_id} ===\n"

        except Exception as e:
            print(f"Error during deep research: {e}")
            traceback.print_exc()
            yield f"Deep research failed for query='{query}'. Please try again later."
