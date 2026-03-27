"""LLM client abstractions and adapters."""

from .messages import Role, Message, ToolCall, ToolResult
from .base import BaseLLMClient
from .ollama import OllamaClient

__all__ = [
    "Role",
    "Message",
    "ToolCall",
    "ToolResult",
    "BaseLLMClient",
    "OllamaClient",
]
