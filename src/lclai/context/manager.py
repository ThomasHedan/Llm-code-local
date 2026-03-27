"""Context window and token budget management for lclai."""

from __future__ import annotations

from typing import Any

from ..llm.messages import Message, Role


CHARS_PER_TOKEN = 4  # Rough approximation: 1 token ~= 4 characters


class ContextManager:
    """Manages conversation history and token budgeting.

    Keeps track of messages and trims older entries when the token budget
    is exceeded, while always preserving the system prompt and the most
    recent messages.
    """

    # Always keep the most recent N messages untouched during trimming
    MIN_RECENT_MESSAGES = 4

    def __init__(
        self,
        max_tokens: int = 8000,
        reserve_output: int = 2000,
    ) -> None:
        """Initialize the context manager.

        Args:
            max_tokens: Total token budget for the context window.
            reserve_output: Tokens to reserve for the model's output.
        """
        self.max_tokens = max_tokens
        self.reserve_output = reserve_output
        self._messages: list[Message] = []
        self._system_prompt: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def input_budget(self) -> int:
        """Total tokens available for input (context minus output reserve)."""
        return self.max_tokens - self.reserve_output

    @property
    def system_tokens(self) -> int:
        """Estimated tokens used by the system prompt."""
        return self.estimate_tokens(self._system_prompt)

    @property
    def messages_tokens(self) -> int:
        """Estimated tokens used by the current message history."""
        total = 0
        for msg in self._messages:
            total += self.estimate_tokens(msg.content)
            for tc in msg.tool_calls:
                import json
                total += self.estimate_tokens(tc.name + json.dumps(tc.arguments))
        return total

    @property
    def used_tokens(self) -> int:
        """Total estimated tokens currently in use."""
        return self.system_tokens + self.messages_tokens

    @property
    def available_tokens(self) -> int:
        """Tokens remaining for new input before trimming is needed."""
        return self.input_budget - self.used_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_system(self, system: str) -> None:
        """Set the system prompt.

        Args:
            system: System prompt text.
        """
        self._system_prompt = system

    def get_system(self) -> str:
        """Return the current system prompt."""
        return self._system_prompt

    def add_message(self, msg: Message) -> None:
        """Append a message to the conversation history.

        Automatically trims old messages if the budget is exceeded.

        Args:
            msg: Message to add.
        """
        self._messages.append(msg)
        self.trim_to_fit()

    def get_messages(self) -> list[Message]:
        """Return the full message history (without the system prompt).

        Returns:
            List of messages in conversation order.
        """
        return list(self._messages)

    def get_all_messages(self) -> list[Message]:
        """Return messages including the system prompt as the first message.

        Returns:
            List starting with system message (if set), then conversation history.
        """
        all_msgs: list[Message] = []
        if self._system_prompt:
            all_msgs.append(Message.system(self._system_prompt))
        all_msgs.extend(self._messages)
        return all_msgs

    def trim_to_fit(self) -> None:
        """Remove oldest non-recent messages to stay within the token budget.

        Strategy:
        - System prompt: always kept.
        - Last MIN_RECENT_MESSAGES messages: always kept.
        - Older messages: trimmed from oldest first until within budget.
        - Long tool results in trimmed range: truncated to save space.
        """
        if self.used_tokens <= self.input_budget:
            return  # Already within budget

        # We can only trim messages older than the last MIN_RECENT_MESSAGES
        safe_count = self.MIN_RECENT_MESSAGES
        trimable_count = len(self._messages) - safe_count

        if trimable_count <= 0:
            # Can't trim further — just truncate large tool results in recent msgs
            self._truncate_tool_results()
            return

        # Remove from oldest trimable messages first
        removed = 0
        while removed < trimable_count and self.used_tokens > self.input_budget:
            self._messages.pop(0)
            removed += 1

        # If still over budget, try truncating long tool results
        if self.used_tokens > self.input_budget:
            self._truncate_tool_results()

    def _truncate_tool_results(self) -> None:
        """Truncate long tool result messages to reduce token usage."""
        max_tool_chars = 500  # Keep at most this many chars per tool result
        for i, msg in enumerate(self._messages):
            if msg.role == Role.TOOL and len(msg.content) > max_tool_chars:
                truncated_content = (
                    msg.content[:max_tool_chars]
                    + f"\n[... truncated, {len(msg.content) - max_tool_chars} chars omitted ...]"
                )
                # Replace with new message (dataclasses are mutable)
                self._messages[i] = Message(
                    role=msg.role,
                    content=truncated_content,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a string.

        Uses ~4 characters per token as a rough approximation.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count (at least 0).
        """
        if not text:
            return 0
        return max(0, len(text) // CHARS_PER_TOKEN)

    def clear(self) -> None:
        """Clear all conversation messages (keeps system prompt)."""
        self._messages.clear()

    def reset(self) -> None:
        """Clear everything including system prompt."""
        self._messages.clear()
        self._system_prompt = ""

    def token_summary(self) -> dict[str, int]:
        """Return a summary of current token usage.

        Returns:
            Dict with keys: max_tokens, reserve_output, input_budget,
            system_tokens, messages_tokens, used_tokens, available_tokens.
        """
        return {
            "max_tokens": self.max_tokens,
            "reserve_output": self.reserve_output,
            "input_budget": self.input_budget,
            "system_tokens": self.system_tokens,
            "messages_tokens": self.messages_tokens,
            "used_tokens": self.used_tokens,
            "available_tokens": self.available_tokens,
        }

    def __len__(self) -> int:
        """Return number of messages in history."""
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ContextManager("
            f"messages={len(self._messages)}, "
            f"used_tokens={self.used_tokens}/{self.input_budget})"
        )
