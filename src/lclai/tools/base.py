"""Base classes for lclai tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from executing a tool."""

    content: str
    is_error: bool = False

    @classmethod
    def success(cls, content: str) -> "ToolResult":
        """Create a successful result."""
        return cls(content=content, is_error=False)

    @classmethod
    def error(cls, content: str) -> "ToolResult":
        """Create an error result."""
        return cls(content=content, is_error=True)


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str
    description: str

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return OpenAI function calling schema for this tool.

        Returns:
            Dict with 'type', 'function' keys in OpenAI format.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            ToolResult with content and error status.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
