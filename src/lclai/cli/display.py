"""Rich UI components for lclai."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.text import Text

from ..__init__ import __version__

# Shared console instance — use stderr=False so output can be piped
console = Console(highlight=False)
error_console = Console(stderr=True, highlight=False)

# Color palette
COLOR_ASSISTANT = "blue"
COLOR_TOOL = "yellow"
COLOR_SUCCESS = "green"
COLOR_ERROR = "red"
COLOR_INFO = "dim"
COLOR_USER = "cyan"
COLOR_SYSTEM = "magenta"


class Display:
    """Collection of Rich-based display helpers for the lclai REPL."""

    def __init__(self, rich_console: Console | None = None) -> None:
        self.console = rich_console or console

    # ------------------------------------------------------------------
    # Welcome / banner
    # ------------------------------------------------------------------

    def print_welcome(self, model: str, base_url: str) -> None:
        """Print the welcome banner with version and model information.

        Args:
            model: Current Ollama model name.
            base_url: Ollama API base URL.
        """
        banner_text = Text()
        banner_text.append("lclai", style=f"bold {COLOR_ASSISTANT}")
        banner_text.append(f" v{__version__}", style="dim")
        banner_text.append("  —  100% local AI code assistant\n", style="")
        banner_text.append(f"Model: ", style="dim")
        banner_text.append(model, style=f"bold {COLOR_TOOL}")
        banner_text.append(f"   API: {base_url}", style="dim")

        self.console.print(
            Panel(
                banner_text,
                border_style=COLOR_ASSISTANT,
                padding=(0, 1),
            )
        )
        self.console.print(
            "Type [bold]/help[/bold] for commands, "
            "[bold]/exit[/bold] to quit, or just start chatting.",
            style="dim",
        )
        self.console.print()

    def print_help(self) -> None:
        """Print available slash commands."""
        commands = [
            ("/help", "Show this help message"),
            ("/clear", "Clear the conversation history"),
            ("/model <name>", "Switch to a different Ollama model"),
            ("/models", "List available Ollama models"),
            ("/context", "Show token usage statistics"),
            ("/exit, /quit", "Exit lclai"),
        ]
        lines = [f"  [bold {COLOR_TOOL}]{cmd:<20}[/] {desc}" for cmd, desc in commands]
        content = "\n".join(lines)
        self.console.print(
            Panel(content, title="Available Commands", border_style=COLOR_INFO, padding=(0, 1))
        )

    # ------------------------------------------------------------------
    # Message display
    # ------------------------------------------------------------------

    def print_user_message(self, text: str) -> None:
        """Echo the user's input (useful when re-displaying after processing).

        Args:
            text: User input text.
        """
        # Usually we don't re-echo since the user typed it, but useful for logs
        label = Text("You: ", style=f"bold {COLOR_USER}")
        self.console.print(label, end="")
        self.console.print(text)

    def begin_assistant_response(self) -> None:
        """Print the assistant label before streaming begins."""
        label = Text("Assistant: ", style=f"bold {COLOR_ASSISTANT}")
        self.console.print(label, end="")

    def print_assistant_chunk(self, chunk: str) -> None:
        """Print a single streaming chunk from the assistant.

        Args:
            chunk: Text chunk to print (no newline added).
        """
        print(chunk, end="", flush=True)

    def end_assistant_response(self, text: str = "") -> None:
        """Finalize the assistant response display.

        If the full text is available (non-streaming), render it as Markdown.
        Otherwise just ensure we're on a new line.

        Args:
            text: Full accumulated response text (empty for streaming-only display).
        """
        if text:
            # Render full response as Markdown
            label = Text("Assistant: ", style=f"bold {COLOR_ASSISTANT}")
            self.console.print(label)
            self.console.print(Markdown(text))
        else:
            # Just move to a new line after streaming
            self.console.print()

    def print_assistant_response(self, text: str) -> None:
        """Print a complete (non-streamed) assistant response with Markdown rendering.

        Args:
            text: Full assistant response text.
        """
        self.console.print()
        label = Text("Assistant", style=f"bold {COLOR_ASSISTANT}")
        self.console.print(Rule(title=label, style=COLOR_ASSISTANT, align="left"))
        self.console.print(Markdown(text))
        self.console.print()

    # ------------------------------------------------------------------
    # Tool display
    # ------------------------------------------------------------------

    def print_tool_use(self, name: str, params: dict[str, Any]) -> None:
        """Display a tool being invoked.

        Args:
            name: Tool name.
            params: Arguments passed to the tool.
        """
        # Format params: truncate long values
        formatted_params = {}
        for k, v in params.items():
            v_str = str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "..."
            formatted_params[k] = v_str

        param_str = ", ".join(f"{k}={v!r}" for k, v in formatted_params.items())
        text = Text()
        text.append("  Tool: ", style=f"bold {COLOR_TOOL}")
        text.append(name, style=f"bold {COLOR_TOOL}")
        text.append(f"({param_str})", style="dim")
        self.console.print(text)

    def print_tool_result(self, name: str, result_content: str, is_error: bool) -> None:
        """Display the result of a tool execution.

        Args:
            name: Tool name.
            result_content: Output from the tool.
            is_error: Whether this is an error result.
        """
        color = COLOR_ERROR if is_error else COLOR_SUCCESS
        prefix = "Error" if is_error else "Result"

        # Truncate long results in display (full content still sent to LLM)
        display_content = result_content
        max_display = 500
        if len(display_content) > max_display:
            display_content = display_content[:max_display] + f"\n  ... ({len(result_content) - max_display} more chars)"

        text = Text()
        text.append(f"  {prefix}[{name}]: ", style=f"bold {color}")
        text.append(display_content, style="dim")
        self.console.print(text)

    # ------------------------------------------------------------------
    # Info / error / status
    # ------------------------------------------------------------------

    def print_error(self, msg: str) -> None:
        """Print an error message in red.

        Args:
            msg: Error message.
        """
        self.console.print(f"[bold {COLOR_ERROR}]Error:[/] {msg}")

    def print_info(self, msg: str) -> None:
        """Print a dimmed informational message.

        Args:
            msg: Info message.
        """
        self.console.print(msg, style=COLOR_INFO)

    def print_success(self, msg: str) -> None:
        """Print a success message in green.

        Args:
            msg: Success message.
        """
        self.console.print(f"[bold {COLOR_SUCCESS}]{msg}[/]")

    def print_warning(self, msg: str) -> None:
        """Print a warning message in yellow.

        Args:
            msg: Warning message.
        """
        self.console.print(f"[bold {COLOR_TOOL}]Warning:[/] {msg}")

    def print_context_stats(self, stats: dict[str, int]) -> None:
        """Display token usage statistics.

        Args:
            stats: Dict from ContextManager.token_summary().
        """
        used = stats["used_tokens"]
        budget = stats["input_budget"]
        pct = int(used / budget * 100) if budget > 0 else 0

        color = COLOR_SUCCESS
        if pct > 80:
            color = COLOR_ERROR
        elif pct > 60:
            color = COLOR_TOOL

        lines = [
            f"  Max context:     {stats['max_tokens']:>8} tokens",
            f"  Reserved output: {stats['reserve_output']:>8} tokens",
            f"  Input budget:    {budget:>8} tokens",
            f"  System prompt:   {stats['system_tokens']:>8} tokens",
            f"  Message history: {stats['messages_tokens']:>8} tokens",
            f"  Total used:      {used:>8} tokens  [{color}]({pct}%)[/{color}]",
            f"  Available:       {stats['available_tokens']:>8} tokens",
        ]
        content = "\n".join(lines)
        self.console.print(
            Panel(content, title="Token Usage", border_style=COLOR_INFO, padding=(0, 1))
        )

    def print_models(self, models: list[str], current: str) -> None:
        """Display available models with the current one highlighted.

        Args:
            models: List of model names.
            current: Currently active model name.
        """
        if not models:
            self.print_info("No models found. Pull a model with: ollama pull <model>")
            return

        lines = []
        for m in models:
            if m == current:
                lines.append(f"  [bold {COLOR_ASSISTANT}]* {m}[/] [dim](current)[/dim]")
            else:
                lines.append(f"    {m}")

        content = "\n".join(lines)
        self.console.print(
            Panel(content, title="Available Models", border_style=COLOR_INFO, padding=(0, 1))
        )

    def print_separator(self) -> None:
        """Print a horizontal separator line."""
        self.console.print(Rule(style=COLOR_INFO))
