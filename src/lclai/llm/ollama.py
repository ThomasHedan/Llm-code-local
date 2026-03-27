"""Ollama LLM client adapter."""

from __future__ import annotations

import json
import sys
from typing import Any, Iterator

import httpx

from .base import BaseLLMClient
from .messages import Message, Role, ToolCall


class OllamaError(Exception):
    """Error communicating with Ollama."""


class OllamaClient(BaseLLMClient):
    """Client for Ollama REST API.

    Supports streaming and tool/function calling using OpenAI-compatible format.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    TIMEOUT_SECONDS = 120.0
    CONNECT_TIMEOUT = 5.0

    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL) -> None:
        super().__init__(model, base_url)
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(self.TIMEOUT_SECONDS, connect=self.CONNECT_TIMEOUT),
        )

    def _check_connection(self) -> None:
        """Verify Ollama is reachable, raise descriptive error if not."""
        try:
            resp = self._client.get("/api/tags", timeout=self.CONNECT_TIMEOUT)
            resp.raise_for_status()
        except httpx.ConnectError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except httpx.TimeoutException:
            raise OllamaError(
                f"Timeout connecting to Ollama at {self._base_url}. "
                "Ollama may be starting up — try again in a moment."
            )
        except httpx.HTTPStatusError as e:
            raise OllamaError(f"Ollama API error: {e.response.status_code} {e.response.text}")

    def _messages_to_ollama(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert our Message objects to Ollama API format."""
        result = []
        for msg in messages:
            result.append(msg.to_dict())
        return result

    def _stream_chat(
        self,
        payload: dict[str, Any],
        on_chunk: Any | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Stream a chat completion and return accumulated text and tool calls.

        Args:
            payload: Request payload for /api/chat.
            on_chunk: Optional callback called with each text chunk.

        Returns:
            Tuple of (accumulated_text, tool_calls).
        """
        accumulated_text = ""
        tool_calls: list[ToolCall] = []

        with self._client.stream("POST", "/api/chat", json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = response.read().decode()
                raise OllamaError(f"Ollama chat error {e.response.status_code}: {body}")

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = chunk.get("message", {})
                content = message.get("content", "")
                if content:
                    accumulated_text += content
                    if on_chunk:
                        on_chunk(content)
                    else:
                        print(content, end="", flush=True)

                # Collect tool calls from streaming chunks
                chunk_tools = message.get("tool_calls", [])
                if chunk_tools:
                    for tc_data in chunk_tools:
                        tool_calls.append(ToolCall.from_dict(tc_data))

                if chunk.get("done", False):
                    break

        return accumulated_text, tool_calls

    def _non_stream_chat(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, list[ToolCall]]:
        """Perform non-streaming chat completion.

        Returns:
            Tuple of (text, tool_calls).
        """
        payload = {**payload, "stream": False}
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"Ollama chat error {e.response.status_code}: {e.response.text}"
            )

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])
        tool_calls = [ToolCall.from_dict(tc) for tc in raw_tool_calls]
        return content, tool_calls

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> Message:
        """Send messages to Ollama and return the assistant response.

        If stream=True, text chunks are printed to stdout as they arrive.
        Tool calls are always collected silently.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages_to_ollama(messages),
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        try:
            if stream:
                text, tool_calls = self._stream_chat(payload)
            else:
                text, tool_calls = self._non_stream_chat(payload)
        except httpx.ConnectError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )

        return Message.assistant(content=text, tool_calls=tool_calls)

    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count using character-based approximation.

        Uses ~4 characters per token as a rough heuristic.
        """
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.content)
            for tc in msg.tool_calls:
                total_chars += len(tc.name)
                total_chars += len(json.dumps(tc.arguments))
        # Add overhead for message formatting
        total_chars += len(messages) * 10
        return max(1, total_chars // 4)

    def list_models(self) -> list[str]:
        """List available Ollama models.

        Returns:
            Sorted list of model names.

        Raises:
            OllamaError: If Ollama is not running or returns an error.
        """
        try:
            response = self._client.get("/api/tags", timeout=self.CONNECT_TIMEOUT)
            response.raise_for_status()
        except httpx.ConnectError:
            raise OllamaError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except httpx.TimeoutException:
            raise OllamaError("Timeout listing models from Ollama.")
        except httpx.HTTPStatusError as e:
            raise OllamaError(f"Error listing models: {e.response.status_code}")

        data = response.json()
        models = data.get("models", [])
        return sorted(m["name"] for m in models if "name" in m)

    def __del__(self) -> None:
        """Clean up HTTP client on deletion."""
        try:
            self._client.close()
        except Exception:
            pass
