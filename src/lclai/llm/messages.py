"""Message types for LLM communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Roles for messages in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Ollama/OpenAI function calling format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Parse from Ollama/OpenAI function calling format."""
        import json

        func = data.get("function", {})
        arguments = func.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        return cls(
            id=data.get("id", ""),
            name=func.get("name", ""),
            arguments=arguments,
        )


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to Ollama/OpenAI API format."""
        d: dict[str, Any] = {"role": self.role.value}

        if self.role == Role.TOOL:
            d["content"] = self.content
            if self.tool_call_id:
                d["tool_call_id"] = self.tool_call_id
            if self.name:
                d["name"] = self.name
        elif self.tool_calls:
            d["content"] = self.content or ""
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        else:
            d["content"] = self.content

        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> "Message":
        """Create an assistant message."""
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def tool_result(cls, result: ToolResult) -> "Message":
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )

    def has_tool_calls(self) -> bool:
        """Check if this message contains tool calls."""
        return bool(self.tool_calls)

    def __str__(self) -> str:
        role_str = self.role.value
        if self.tool_calls:
            names = ", ".join(tc.name for tc in self.tool_calls)
            return f"[{role_str}] (tool_calls: {names}) {self.content[:100]}"
        return f"[{role_str}] {self.content[:100]}"
