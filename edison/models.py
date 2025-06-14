"""
Models module for Edison API configurations and agent outputs.

This module contains Pydantic models used for configuration management
and structuring agent outputs in the Edison system.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type


class EdisonApiKeyConfig(BaseModel):
    """Configuration for storing API keys required by Edison.

    This class manages the API keys needed for various services used by Edison,
    including OpenAI, Firecrawl, and Serper.

    Attributes:
        openai_api_key (str): API key for OpenAI services.
        firecrawl_api_key (str): API key for Firecrawl services.
        serper_api_key (str): API key for Serper services.
    """

    openai_api_key: str
    firecrawl_api_key: str
    serper_api_key: str


class ExpanderAgentOutput(BaseModel):
    """Output container for the Expander Agent's results.

    This class structures the output data produced by the Expander Agent,
    primarily storing related search queries.

    Attributes:
        related_queries (List[str]): List of related search queries generated
            by the Expander Agent.
    """

    related_queries: List[str]


class QnaItem(BaseModel):
    """Container for a single QnA item.

    This class represents a single question and its corresponding answer.

    Attributes:
        question (str): The question text.
        answer (str): The answer text.
    """

    question: str
    answer: str


class ExpandedQnaItem(BaseModel):
    """Container for an expanded QnA item.

    This class represents a single question and its corresponding answer,
    along with additional context.

    Attributes:
        question (str): The question text.
        answer (str): The answer text.
        context (str): Additional context related to the question and answer.
    """

    qna_pair: QnaItem
    expansion: str


class QnaAgentOutput(BaseModel):
    """Output container for the Questioning Agent's results.

    This class structures the output data produced by the Questioning Agent,
    primarily storing the generated questions.

    Attributes:
        questions (List[str]): List of questions generated by the Questioning Agent.
    """

    qna_pairs: List[QnaItem]


class DocumentSection(BaseModel):
    """A section within a document."""

    title: str
    content: str
    version: int = 0
    last_modified: Optional[datetime] = Field(default_factory=datetime.now)
    context_tokens: Optional[int] = None


class DocumentMetdataItem(BaseModel):
    """Container for document metadata item.

    This class represents a single metadata item associated with a document.

    Attributes:
        key (str): The key of the metadata item.
        value (str): The value of the metadata item.
    """

    key: str
    value: str


class DocumentContent(BaseModel):
    """Content of a document including sections."""

    sections: Dict[str, DocumentSection] = Field(default_factory=dict)
    metadata: List[DocumentMetdataItem] = Field(default_factory=list)
    version: int = 0
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    last_modified: Optional[datetime] = Field(default_factory=datetime.now)


class CreateDocumentArgs(BaseModel):
    """Container for document creation arguments.

    This class defines the required parameters for creating a new document
    in the system, including document ID and metadata.

    Attributes:
        doc_id (str): Unique identifier for the document.
        metadata (Dict[str, str]): Document metadata key-value pairs.
        storage_dir (str): Directory for document storage, defaults to "documents".
    """

    doc_id: str
    # metadata: List[DocumentMetdataItem]

    class Config:
        extra = "forbid"


class UpdateSectionArgs(BaseModel):
    """Container for section update arguments.

    This class defines the required parameters for updating an existing
    section within a document.

    Attributes:
        doc_id (str): Identifier of the document containing the section.
        section_id (str): Identifier of the section to update.
        title (str): New title for the section.
        content (str): New content for the section.
        context_tokens (int): Number of context tokens in the content.
        storage_dir (str): Directory for document storage, defaults to "documents".
    """

    doc_id: str
    title: str
    content: str

    class Config:
        extra = "forbid"


class OrganizeSectionsArgs(BaseModel):
    """Container for section organization arguments.

    This class defines the parameters for organizing document sections,
    including token limit constraints.

    Attributes:
        doc_id (str): Identifier of the document to organize.
        max_tokens (int): Maximum tokens per section, defaults to 2048.
        storage_dir (str): Directory for document storage, defaults to "documents".
    """

    doc_id: str

    class Config:
        extra = "forbid"


class ListDocumentsArgs(BaseModel):
    """Container for document listing arguments.

    This class defines the parameters for listing documents from storage.

    Attributes:
        storage_dir (str): Directory to list documents from, defaults to "documents".
    """

    class Config:
        extra = "forbid"


class ToolType(Enum):
    """Enumeration of available tool types in the Edison system.

    Each tool type represents a specialized function with specific capabilities:
        UPDATE_SECTION: Updates or creates document sections
        WEB_SEARCH: Performs web searches for information retrieval
    """

    UPDATE_SECTION = "update_section"
    WEB_SEARCH = "web_search"


class AgentType(Enum):
    """Enumeration of available agent types in the Edison system.

    Each agent type represents a specialized AI agent with specific capabilities:
        TASKS_AGENT: Executes specific research tasks
        QNA_AGENT: Generates and answers questions
        SUMMARIZER_AGENT: Provides concise summaries
        GENERATOR_AGENT: Creates content and information
        QUERY_EXPANDER_AGENT: Expands search queries for broader coverage
        DOCUMENT_WRITER_AGENT: Manages document content with versioning and organization
    """

    TASKS_AGENT = "tasks_agent"
    QNA_AGENT = "qna_agent"
    SUMMARIZER_AGENT = "summarizer_agent"
    GENERATOR_AGENT = "generator_agent"
    QUERY_EXPANDER_AGENT = "query_expander_agent"
    DOCUMENT_WRITER_AGENT = "document_writer_agent"
    ORCHESTRATOR_AGENT = "orchestrator_agent"


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
    description: str
    instructions: str
    model: str
    tools: Optional[List[Any]] = None
    output_type: Optional[Type] = None
    handoffs: Optional[List[AgentType]] = None
    agent_tools: Optional[List[AgentType]] = None


class ComparisonResult(BaseModel):
    """Result of comparing two sections."""

    similarity_score: float = Field(..., description="Similarity score between 0 and 1")
    explanation: str = Field(..., description="Explanation of the similarity")


class MergeResult(BaseModel):
    """Result of merging two sections."""

    merged_title: str = Field(..., description="The merged section title")
    merged_content: str = Field(..., description="The merged section content")
    source_sections: list[str] = Field(
        default_factory=list, description="IDs of sections that were merged"
    )
