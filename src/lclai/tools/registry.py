"""Tool registry for lclai."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseTool, ToolResult
from .file_read import ReadFileTool
from .file_write import WriteFileTool
from .file_edit import EditFileTool
from .bash import BashTool
from .glob_search import GlobSearchTool
from .grep_search import GrepSearchTool

if TYPE_CHECKING:
    from ..doc_reader.indexer import DocIndex


class ToolRegistry:
    """Registry that holds and manages all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its name.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool {tool.name!r} is already registered. "
                "Use a different name or remove the existing tool first."
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove.
        """
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            The tool, or None if not found.
        """
        return self._tools.get(name)

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function calling schemas for all registered tools.

        Returns:
            List of tool schemas suitable for passing to an LLM.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Args:
            name: Name of the tool to execute.
            **kwargs: Arguments to pass to the tool's execute method.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys()))
            return ToolResult.error(
                f"Unknown tool: {name!r}. Available tools: {available}"
            )

        try:
            return tool.execute(**kwargs)
        except TypeError as e:
            return ToolResult.error(
                f"Invalid arguments for tool {name!r}: {e}"
            )
        except Exception as e:
            return ToolResult.error(
                f"Tool {name!r} raised an unexpected error: {type(e).__name__}: {e}"
            )

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._tools.keys()))
        return f"ToolRegistry([{names}])"


def get_default_tools(doc_index: "DocIndex | None" = None) -> ToolRegistry:
    """Create and return a ToolRegistry pre-loaded with all default tools.

    Args:
        doc_index: Optional DocIndex instance. If provided, the DocExamplesTool
            is registered so the LLM can query indexed documentation examples.

    Returns:
        ToolRegistry with all built-in tools registered.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(BashTool())
    registry.register(GlobSearchTool())
    registry.register(GrepSearchTool())

    if doc_index is not None:
        from .doc_examples import DocExamplesTool
        registry.register(DocExamplesTool(doc_index))

    return registry
