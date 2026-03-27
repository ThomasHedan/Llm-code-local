"""Abstract base class for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .messages import Message


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: str) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        """Return the current model name."""
        return self._model

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set the current model name."""
        self._model = value

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> Message:
        """Send messages and return assistant response.

        Args:
            messages: Conversation history.
            tools: Optional list of tool schemas in OpenAI function calling format.
            stream: Whether to stream the response (prints chunks to stdout).

        Returns:
            The assistant's response as a Message.
        """
        ...

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a list of messages.

        Args:
            messages: Messages to count tokens for.

        Returns:
            Estimated token count.
        """
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names.
        """
        ...
