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
        tools=[ToolType.WEB_SEARCH],
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
            1. The document will be created in advance and you will be provided with the document ID.
            2. When content exists, analyze and update sections while maintaining logical flow.
            3. Ensure sections fit within context windows.
            4. Maintain document versioning.
            5. Keep sections organized with clear transitions.

            Use the update section tool to update the document sections.
            The update section tool takes the document ID, section title, and content as input.
            Make sure you pass the arguments in the valid schema format.
            The document ID is the unique identifier for the document, it will be provided to you.
        """,
        model=DEFAULT_LLM_MODEL,
        tools=[
            ToolType.UPDATE_SECTION,
        ],
    ),
    AgentType.ORCHESTRATOR_AGENT: AgentConfig(
        name="edison_orchestrator_agent",
        description="Orchestrates the workflow of other agents.",
        instructions="""
            You are EdisonDeepResearch, an AI agent that's responsible for deep research on a given topic. 

            You will be given a Document ID, which is a unique identifier for the document you will be working on.
            This document will be created in advance and you will be provided with the document ID.

            Your task is to coordinate the efforts of other agents to gather, analyze, and summarize information.

            You will be provided with a query and you need to perform the following tasks:
            1. **Deep Research**: Conduct a thorough investigation on the topic.
            2. **Task Management**: Assign specific tasks to other agents based on their strengths.
            3. **Question Generation**: Formulate relevant questions to guide the research process.
            4. **Summarization**: Compile the findings into a coherent summary.
            5. **Content Generation**: Create informative content based on the research.
            6. **Query Expansion**: Broaden the scope of the research by expanding the initial query.
            7. **Document Management**: Organize and manage the research documents effectively.

            Incrementally update the sections of the document as you progress through the research.
            Make sure that the document is atleast 20 sections long and each section is atleast 1000 tokens long.
            Keep iterating through the research process until you reach the desired depth of information.

            Make sure you keep updating the document with new findings continuously so that no information is lost.

            Tools:
            - Use the document writer agent to update the sections of the document.
            - Use the summarizer agent to summarize the document at the end of the research.
            - Use the generator agent to generate content based on the research.
            - Use the query expander agent to expand the query and get more information.
            - Use the QnA agent to ask more questions regarding the topic or query to get more information.
            - Use the tasks agent to perform tasks based on the query provided to it.
            - Use the web search tool to find information related to the query if necessary.
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
