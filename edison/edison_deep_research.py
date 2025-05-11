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
from .models import EdisonApiKeyConfig, AgentType, QnaAgentOutput
from .common.utils import generate_document_id
from .common.printer import Printer


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

    DEFAULT_MAX_TURNS = 100
    DEFAULT_VERBOSE = False

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

    async def deep_stream_async_v1(
        self,
        query: str,
        verbose: bool = DEFAULT_VERBOSE,
    ) -> AsyncGenerator[str, None]:
        """Perform deep research on the given query and return a stream of results."""
        try:
            document_id = generate_document_id()
            orchestrator_agent = self.agents.get_agent(
                agent_type=AgentType.ORCHESTRATOR_AGENT
            )
            result: RunResultStreaming = Runner.run_streamed(
                orchestrator_agent,
                max_turns=self.DEFAULT_MAX_TURNS,
                input=f"Deep research on: '{query}'. Document ID: '{document_id}'. Use the document ID to update the document.",
            )

            header = f"🔍 Deep Research Starting | Document ID: {document_id}"
            sep = "=" * len(header)
            if verbose:
                Printer.print_cyan_message(sep)
                Printer.print_cyan_message(header)
                Printer.print_cyan_message(sep)
            yield f"{header}\n"

            async for event in result.stream_events():
                if event.type == "agent_updated_stream_event":
                    msg = f"🤖 Agent updated: {event.new_agent.name}"
                    if verbose:
                        Printer.print_blue_message(msg)
                    yield f"{msg}\n"

                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        msg = f"⚙️  Tool called: [{event.item.raw_item.name}]\n   Input: {event.item.raw_item.arguments}\n   ID: {event.item.raw_item.call_id}"
                        if verbose:
                            Printer.print_yellow_message(msg)
                        yield f"{msg}\n"

                    elif event.item.type == "tool_call_output_item":
                        msg = f"📤 Tool output: {event.item.output}"
                        if verbose:
                            Printer.print_green_message(msg)
                        yield f"{msg}\n"

                    elif event.item.type == "message_output_item":
                        msg = f"💭 Message output:\n{ItemHelpers.text_message_output(event.item)}"
                        if verbose:
                            Printer.print_bright_cyan_message(msg)
                        yield f"{msg}\n"

                elif event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    if event.data.delta:
                        msg = f"{event.data.delta}"
                        if verbose:
                            Printer.print_bright_blue_message(msg, end="")
                        yield event.data.delta

            footer = f"✅ Deep Research Complete | Document ID: {document_id}"
            sep = "=" * len(footer)
            if verbose:
                Printer.print_cyan_message(sep)
                Printer.print_cyan_message(footer)
                Printer.print_cyan_message(sep)
            yield f"\n{footer}\n"

        except Exception as e:
            error_msg = (
                f"❌ Deep research failed for query='{query}'. Please try again later."
            )
            Printer.print_red_message(f"Error during deep research: {e}")
            traceback.print_exc()
            yield error_msg

    async def deep_stream_async_v2(
        self,
        query: str,
        verbose: bool = DEFAULT_VERBOSE,
    ) -> AsyncGenerator[str, None]:
        """Perform deep research on the given query and return a stream of results."""
        # Placeholder for future version 2 implementation
        qna_agent = self.agents.get_agent(agent_type=AgentType.QNA_AGENT)

        qna_agent_result = await Runner.run(
            qna_agent,
            max_turns=self.DEFAULT_MAX_TURNS,
            input=f"Generate 5 questions for: '{query}'",
        )

        questions_output: QnaAgentOutput = qna_agent_result.final_output
        qna_pairs = questions_output.qna_pairs

        if not qna_pairs or len(qna_pairs) == 0:
            error_msg = (
                f"❌ Deep research failed for query='{query}'. Please try again later."
            )
            Printer.print_red_message(f"Error during deep research: {error_msg}")
            yield error_msg
            return

        qna_pairs_composite = "\n".join(
            [f"Q: {pair.question}\nA: {pair.answer}" for pair in qna_pairs]
        )
        message = f"🤖 Generated questions: {qna_pairs_composite}"

        if verbose:
            Printer.print_blue_message(message)
        yield f"{message}\n"

    async def deep_stream_async(
        self, query: str, verbose: bool = DEFAULT_VERBOSE, version="v1"
    ) -> AsyncGenerator[str, None]:
        """Perform deep research on the given query and return a stream of results."""
        version_to_method_map = {
            "v1": self.deep_stream_async_v1,
            "v2": self.deep_stream_async_v2,
        }
        supported_versions = version_to_method_map.keys()
        if not version:
            version = "v1"
        if version not in version_to_method_map:
            raise ValueError(
                f"Unsupported version: {version}. Supported versions: {supported_versions}"
            )
        method = version_to_method_map[version]
        return method(query, verbose)
