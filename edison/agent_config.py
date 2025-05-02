"""Agent Configuration Module.

This module provides configuration settings for specialized AI agents used in deep research operations.
It defines default models and configuration parameters for each agent type, including their
instructions, tools, and output types.

Typical usage example:
    agent_config = AGENT_CONFIGS[AgentType.TASKS_AGENT]
    agent = Agent(
        name=agent_config.name,
        instructions=agent_config.instructions,
        model=agent_config.model
    )

Note:
    Configuration changes here will affect the behavior of all Edison agents.
    Make sure to test changes thoroughly before deployment.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

from .models import ExpanderAgentOutput, QnaAgentOutput
from .models import ToolType, AgentType, AgentConfig
from typing import Dict


DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_QNA_MODEL = "gpt-4o"


AGENT_CONFIGS: Dict[AgentType, AgentConfig] = {
    AgentType.TASKS_AGENT: AgentConfig(
        name="edison_task_agent",
        description="Performs tasks based on the query provided to it.",
        instructions="""
            You are an AI agent that performs tasks based on the query provided to you.
            You will be provided with a query and you need to perform the task.
        """,
        model=DEFAULT_QNA_MODEL,
    ),
    AgentType.QNA_AGENT: AgentConfig(
        name="edison_qna_agent",
        description="Generates and answers questions based on the query provided to it.",
        instructions="""
            You are an AI agent that asks more questions regarding a topic or query to get more information.
            You will be provided with a query and you need to ask more questions to get more information.

            Use the web search tool to find information related to the query if necessary.
        """,
        model=DEFAULT_LLM_MODEL,
        output_type=QnaAgentOutput,
        tools=[ToolType.WEB_SEARCH],
    ),
    AgentType.SUMMARIZER_AGENT: AgentConfig(
        name="edison_summarizer_agent",
        description="Summarizes the information provided to it.",
        instructions="""
            You are an AI agent that summarizes the information provided to you.
            You will be provided with a query and you need to summarize the information.
        """,
        model=DEFAULT_LLM_MODEL,
    ),
    AgentType.GENERATOR_AGENT: AgentConfig(
        name="edison_generator_agent",
        description="Generates information based on the query provided to it.",
        instructions="""
            You are an AI agent that generates information based on the query provided to you.
            You will be provided with a query and you need to generate information.

            Use the web search tool to find information related to the query if necessary.
        """,
        model=DEFAULT_LLM_MODEL,
        tools=[ToolType.WEB_SEARCH],
    ),
    AgentType.QUERY_EXPANDER_AGENT: AgentConfig(
        name="edison_query_expander_agent",
        description="Expands the query provided to it.",
        instructions="""
            You are an AI agent that expands the query provided to you.
            You will be provided with a query and you need to expand it.
        """,
        model=DEFAULT_LLM_MODEL,
        output_type=ExpanderAgentOutput,
    ),
    AgentType.DOCUMENT_WRITER_AGENT: AgentConfig(
        name="edison_document_writer_agent",
        description="Manages document content, handling versioning and organization.",
        instructions="""
            You are an AI agent that manages document content, handling versioning and organization.

            For a given document:
            1. When empty, create appropriate sections and add initial content.
            2. When content exists, analyze and update sections while maintaining logical flow.
            3. Ensure sections fit within context windows.
            4. Maintain document versioning.
            5. Keep sections organized with clear transitions.

            Use the following tools:
            - Create Document: To create a new document.
            - Update Section: To update or create sections within the document.
            - Organize Sections: To reorganize sections within the document while maintaining token limits.
            - List Documents: To list available documents.
        """,
        model=DEFAULT_LLM_MODEL,
        tools=[
            ToolType.CREATE_DOCUMENT,
            ToolType.UPDATE_SECTION,
            ToolType.ORGANIZE_SECTIONS,
            ToolType.LIST_DOCUMENTS,
        ],
    ),
    AgentType.ORCHESTRATOR_AGENT: AgentConfig(
        name="edison_orchestrator_agent",
        description="Orchestrates the workflow of other agents.",
        instructions="""
            You are EdisonDeepResearch AI agent that performs deep research on a given query.
            You will be provided with a query and you need to manage the workflow of other agents.

            - Given a query, expand the query using the Query Expander Agent.
            - Use the expanded query to generate questions using the Questioning Agent.
            - Use the generated questions to perform research using the Generator Agent which uses the web search tool.
            - Use the research results to summarize the information using the Summarizer Agent.
            - Summarize the information if necessary to fit within the context window of the LLM.
            - If you have other mission-critical tasks, use the Tasks Agent to perform them.
            - The document should be at least 5 pages long. Use the Document Writer Agent to create and update the document in sections.
            - Repeat this process until the document is complete.
        """,
        model=DEFAULT_LLM_MODEL,
        agent_tools=[
            AgentType.TASKS_AGENT,
            AgentType.QNA_AGENT,
            AgentType.SUMMARIZER_AGENT,
            AgentType.GENERATOR_AGENT,
            AgentType.QUERY_EXPANDER_AGENT,
            AgentType.DOCUMENT_WRITER_AGENT,
        ],
    ),
}
