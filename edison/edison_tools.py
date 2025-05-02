"""Edison Tools Module.

This module provides a collection of specialized tools used for document management
and manipulation within the Edison system. Each tool is designed for specific
operations such as document creation, section updates, and content organization.

Typical usage example:
    tools = EdisonTools()
    document_tools = tools.get_tools()
    create_doc_tool = document_tools[0]
    result = create_doc_tool.invoke(args)

Note:
    All tools require proper storage directory configuration before use.
    See DocumentWriterTool class for storage configuration details.

Author: Aditya Patange (https://www.github.com/AdiPat)
"""

from typing import Dict, List, Any
from agents import FunctionTool, RunContextWrapper, WebSearchTool
from .tools.document_tools import DocumentWriterTool
from .models import (
    CreateDocumentArgs,
    UpdateSectionArgs,
    OrganizeSectionsArgs,
    ListDocumentsArgs,
    ToolType,
)


class EdisonTools:
    """Manager class for Edison Tools.

    This class handles the organization and access to various specialized tools
    used in Edison operations. It provides methods to access document management
    tools and other utility functions.

    Attributes:
        _tools (List[FunctionTool]): Internal list storing tool instances
    """

    def __init__(self):
        """Initializes the Edison tools collection."""
        self._tools: List[FunctionTool] = self._init_document_tools()
        self._tools_map: Dict[ToolType, FunctionTool] = {
            ToolType.UPDATE_SECTION: self._tools[0],
            ToolType.WEB_SEARCH: self._tools[1],
        }

    def _init_document_tools(self) -> List[FunctionTool]:
        """Initializes document management tools.

        This method creates and configures function tools for document operations
        including creation, updates, organization and listing. Each tool is
        initialized with appropriate handlers and parameter schemas.

        Returns:
            List[FunctionTool]: Collection of initialized document management tools
        """

        async def update_section_handler(ctx: RunContextWrapper[Any], args: str) -> str:
            """Handles updating document sections.

            Updates or creates a section within an existing document with new content
            and metadata while maintaining context token limits.

            Args:
                ctx: Runtime context wrapper containing execution state
                args: JSON string containing section update parameters

            Returns:
                str: Success message with section and document IDs or error message

            Raises:
                Exception: If section update fails
            """
            try:
                parsed = UpdateSectionArgs.model_validate_json(args)
                tool = DocumentWriterTool(storage_dir="documents")
                tool.update_section(
                    doc_id=parsed.doc_id,
                    title=parsed.title,
                    content=parsed.content,
                )
                return f"Updated section {parsed.title} in document {parsed.doc_id}."
            except Exception as e:
                return f"Failed to update section: {str(e)}"

        return [
            FunctionTool(
                name="update_section",
                description="Updates or creates a section in a document",
                params_json_schema=UpdateSectionArgs.model_json_schema(),
                on_invoke_tool=update_section_handler,
                strict_json_schema=True,
            ),
            WebSearchTool(),
        ]

    def get_tools(self) -> List[FunctionTool]:
        """Returns all available Edison tools.

        Provides access to the complete collection of initialized tools
        that can be used for document operations.

        Returns:
            List[FunctionTool]: List of all initialized document management tools
        """
        return self._tools

    def get_tool(self, tool_type: ToolType) -> FunctionTool:
        """Retrieves a specific tool by its type.

        Args:
            tool_type (ToolType): The type of tool to retrieve

        Returns:
            FunctionTool: The requested tool instance

        Raises:
            ValueError: If the tool_type is invalid
        """
        if not isinstance(tool_type, ToolType):
            raise ValueError(f"Invalid tool type: {tool_type}")
        return self._tools_map[tool_type]
