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
            ToolType.CREATE_DOCUMENT: self._tools[0],
            ToolType.UPDATE_SECTION: self._tools[1],
            ToolType.ORGANIZE_SECTIONS: self._tools[2],
            ToolType.LIST_DOCUMENTS: self._tools[3],
            ToolType.WEB_SEARCH: self._tools[4],
        }

    def _init_document_tools(self) -> List[FunctionTool]:
        """Initializes document management tools.

        This method creates and configures function tools for document operations
        including creation, updates, organization and listing. Each tool is
        initialized with appropriate handlers and parameter schemas.

        Returns:
            List[FunctionTool]: Collection of initialized document management tools
        """

        async def create_document_handler(
            ctx: RunContextWrapper[Any], args: CreateDocumentArgs
        ) -> str:
            """Handles creation of new documents.

            Creates a new document with specified metadata in the given storage location.

            Args:
                ctx: Runtime context wrapper containing execution state
                args: JSON string containing document creation parameters

            Returns:
                str: Success message with document ID and metadata or error message

            Raises:
                Exception: If document creation fails
            """
            try:
                parsed = args
                tool = DocumentWriterTool(storage_dir="documents")
                tool.create_document(parsed.doc_id, metadata=None)
                return f"Created document {parsed.doc_id}."
            except Exception as e:
                return f"Failed to create document: {str(e)}"

        async def update_section_handler(
            ctx: RunContextWrapper[Any], args: UpdateSectionArgs
        ) -> str:
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
                parsed = args
                tool = DocumentWriterTool(storage_dir="documents")
                tool.update_section(
                    doc_id=parsed.doc_id,
                    title=parsed.title,
                    content=parsed.content,
                )
                return (
                    f"Updated section {parsed.section_id} in document {parsed.doc_id}."
                )
            except Exception as e:
                return f"Failed to update section: {str(e)}"

        async def organize_sections_handler(
            ctx: RunContextWrapper[Any], args: OrganizeSectionsArgs
        ) -> List[str]:
            """Handles organizing document sections.

            Reorganizes document sections to ensure they fit within specified token limits
            while maintaining logical flow and content integrity.

            Args:
                ctx: Runtime context wrapper containing execution state
                args: JSON string containing organization parameters

            Returns:
                List[str]: List of organized section IDs or error message

            Raises:
                Exception: If section organization fails
            """
            try:
                parsed = args
                tool = DocumentWriterTool(storage_dir="documents")
                return tool.organize_sections(parsed.doc_id, max_tokens=4096)
            except Exception as e:
                return f"Failed to organize sections: {str(e)}"

        async def list_documents_handler(
            ctx: RunContextWrapper[Any], args: ListDocumentsArgs
        ) -> Dict[str, Dict[str, str]]:
            """Handles listing available documents.

            Retrieves a list of all available documents and their associated metadata
            from the specified storage location.

            Args:
                ctx: Runtime context wrapper containing execution state
                args: JSON string containing listing parameters

            Returns:
                Dict[str, Dict[str, str]]: Dictionary of document IDs and their metadata
                                         or error message

            Raises:
                Exception: If document listing fails
            """
            try:
                tool = DocumentWriterTool(storage_dir="documents")
                return tool.list_documents()
            except Exception as e:
                return f"Failed to list documents: {str(e)}"

        return [
            FunctionTool(
                name="create_document",
                description="Creates a new empty document with metadata",
                params_json_schema=CreateDocumentArgs.model_json_schema(),
                on_invoke_tool=create_document_handler,
            ),
            FunctionTool(
                name="update_section",
                description="Updates or creates a section in a document",
                params_json_schema=UpdateSectionArgs.model_json_schema(),
                on_invoke_tool=update_section_handler,
            ),
            FunctionTool(
                name="organize_sections",
                description="Organizes document sections to fit within token limits",
                params_json_schema=OrganizeSectionsArgs.model_json_schema(),
                on_invoke_tool=organize_sections_handler,
            ),
            FunctionTool(
                name="list_documents",
                description="Lists all available documents with their metadata",
                params_json_schema=ListDocumentsArgs.model_json_schema(),
                on_invoke_tool=list_documents_handler,
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
