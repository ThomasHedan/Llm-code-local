"""Main REPL loop for lclai."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console

from ..llm.base import BaseLLMClient
from ..llm.messages import Message, Role
from ..tools.registry import ToolRegistry
from ..context.manager import ContextManager
from .display import Display

# Maximum tool-use iterations per user turn to prevent infinite loops
MAX_TOOL_ITERATIONS = 20

SYSTEM_PROMPT = """\
You are lclai, a local AI code assistant running in a restricted environment.
You help developers write, read, and understand code.

You have access to these tools:
- read_file: Read file contents with line numbers
- write_file: Write or create files
- edit_file: Make precise edits to files (string replacement)
- bash: Execute shell commands
- glob_search: Find files by pattern
- grep_search: Search file contents

Guidelines:
- Read files before editing them
- Make minimal, precise changes
- Prefer edit_file over write_file for modifications
- Always verify your changes work
- Be concise and direct
"""

PROMPT = "> "


class REPL:
    """Interactive REPL for lclai.

    Handles the main input loop, slash commands, and the agentic tool-use loop.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        tools: ToolRegistry,
        context: ContextManager,
        display: Display | None = None,
    ) -> None:
        """Initialize the REPL.

        Args:
            llm: LLM client (e.g. OllamaClient).
            tools: Registry of available tools.
            context: Context/token budget manager.
            display: Display helper (created automatically if None).
        """
        self.llm = llm
        self.tools = tools
        self.context = context
        self.display = display or Display()
        self._running = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the REPL main loop.

        Reads user input, handles slash commands, and processes messages
        until the user exits.
        """
        self._running = True

        # Set up system prompt
        self.context.set_system(SYSTEM_PROMPT)

        # Show welcome banner
        self.display.print_welcome(
            model=self.llm.model_name,
            base_url=self.llm._base_url,
        )

        while self._running:
            try:
                user_input = self._get_input()
            except EOFError:
                # Ctrl+D
                self.display.print_info("\nBye!")
                break
            except KeyboardInterrupt:
                # Ctrl+C at the prompt — just print a newline and continue
                self.display.console.print()
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                handled = self._handle_slash_command(user_input)
                if not handled:
                    self.display.print_error(
                        f"Unknown command: {user_input!r}. Type /help for available commands."
                    )
                continue

            # Process the message through the LLM + tool loop
            try:
                self._process_message(user_input)
            except KeyboardInterrupt:
                # Ctrl+C during generation — cancel cleanly
                self.display.console.print()
                self.display.print_info("[Generation cancelled]")
            except Exception as e:
                self.display.print_error(f"Unexpected error: {e}")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _get_input(self) -> str:
        """Read a line of input from the user.

        Returns:
            Raw input string.

        Raises:
            EOFError: On Ctrl+D.
            KeyboardInterrupt: On Ctrl+C.
        """
        try:
            # Use Rich console.input for a colored prompt when possible
            return self.display.console.input(f"[bold cyan]{PROMPT}[/]")
        except (EOFError, KeyboardInterrupt):
            raise

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _handle_slash_command(self, raw: str) -> bool:
        """Handle a slash command.

        Args:
            raw: Raw input string starting with '/'.

        Returns:
            True if the command was recognized, False otherwise.
        """
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            self.display.print_info("Bye!")
            self._running = False
            return True

        if cmd == "/help":
            self.display.print_help()
            return True

        if cmd == "/clear":
            self.context.clear()
            self.display.print_success("Conversation cleared.")
            return True

        if cmd == "/context":
            stats = self.context.token_summary()
            self.display.print_context_stats(stats)
            return True

        if cmd == "/models":
            self._cmd_list_models()
            return True

        if cmd == "/model":
            self._cmd_switch_model(arg)
            return True

        return False

    def _cmd_list_models(self) -> None:
        """Handle the /models command."""
        try:
            models = self.llm.list_models()
            self.display.print_models(models, self.llm.model_name)
        except Exception as e:
            self.display.print_error(f"Failed to list models: {e}")

    def _cmd_switch_model(self, model_name: str) -> None:
        """Handle the /model <name> command.

        Args:
            model_name: Model name to switch to.
        """
        if not model_name:
            self.display.print_info(f"Current model: [bold]{self.llm.model_name}[/bold]")
            self.display.print_info("Usage: /model <model-name>  (e.g. /model llama3.2:3b)")
            return

        old_model = self.llm.model_name
        self.llm.model_name = model_name
        self.display.print_success(f"Switched model from {old_model!r} to {model_name!r}")
        self.display.print_info(
            "Note: Conversation history is preserved. "
            "Use /clear if the new model needs a fresh context."
        )

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _process_message(self, user_input: str) -> None:
        """Process a user message through the LLM and tool loop.

        Adds the user message to context, sends to LLM, executes any tool
        calls, and streams the final text response.

        Args:
            user_input: The user's raw input text.
        """
        user_msg = Message.user(user_input)
        self.context.add_message(user_msg)

        # Run the agentic loop
        final_message = self._run_tool_loop()

        # Display the final text response
        if final_message.content:
            # We already streamed it, just add newline
            self.display.console.print()
        else:
            self.display.print_info("(No response text)")

    def _run_tool_loop(self) -> Message:
        """Run the agentic tool-use loop.

        Sends messages to the LLM, executes tool calls, and repeats until
        the model produces a response without tool calls (or max iterations).

        Returns:
            The final assistant message (with text content, no tool calls).
        """
        tool_schemas = self.tools.get_all_schemas()
        iteration = 0

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1
            all_messages = self.context.get_all_messages()

            # Determine if we should stream this response
            # Stream when this might be the final text response
            # Don't stream during tool rounds (we want to detect tool calls first)
            # Actually: stream always, detect tool_calls after
            # For simplicity, we stream and collect simultaneously
            streaming = True

            if streaming and iteration == 1:
                # Print the assistant label before the stream starts
                self.display.begin_assistant_response()

            try:
                response = self._call_llm(
                    messages=all_messages,
                    tools=tool_schemas,
                    stream=streaming and not self._will_use_tools_hint(all_messages),
                )
            except Exception as e:
                self.display.print_error(str(e))
                error_msg = Message.assistant(content=f"[Error: {e}]")
                self.context.add_message(error_msg)
                return error_msg

            # Add assistant response to context
            self.context.add_message(response)

            # If no tool calls, we're done
            if not response.has_tool_calls():
                if response.content:
                    # Content was already streamed or needs printing
                    pass
                return response

            # Execute tool calls
            self.display.console.print()  # Newline after any streamed content
            tool_results = self._execute_tool_calls(response)

            # Add tool results to context
            for result_msg in tool_results:
                self.context.add_message(result_msg)

            # Print assistant label again for next iteration
            self.display.begin_assistant_response()

        # Max iterations reached
        self.display.print_warning(
            f"Reached maximum tool iterations ({MAX_TOOL_ITERATIONS}). Stopping."
        )
        return Message.assistant(
            content=f"[Stopped: exceeded {MAX_TOOL_ITERATIONS} tool iterations]"
        )

    def _will_use_tools_hint(self, messages: list[Message]) -> bool:
        """Heuristic: guess if the model is likely to respond with tool calls.

        Used to decide whether to stream or collect the full response first.
        For now, always returns False (always stream), since Ollama supports
        streaming with tool calls in the final chunk.

        Args:
            messages: Current message history.

        Returns:
            True if we suspect tool calls will be used.
        """
        return False

    def _call_llm(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        stream: bool,
    ) -> Message:
        """Call the LLM client with error handling.

        Args:
            messages: Full message list to send.
            tools: Tool schemas for function calling.
            stream: Whether to stream the response.

        Returns:
            Assistant message from the LLM.

        Raises:
            Exception: On LLM communication errors.
        """
        return self.llm.chat(
            messages=messages,
            tools=tools if tools else None,
            stream=stream,
        )

    def _execute_tool_calls(self, assistant_msg: Message) -> list[Message]:
        """Execute all tool calls in an assistant message.

        Args:
            assistant_msg: Assistant message containing tool_calls.

        Returns:
            List of tool result messages to add to context.
        """
        result_messages: list[Message] = []

        for tool_call in assistant_msg.tool_calls:
            # Display the tool invocation
            self.display.print_tool_use(tool_call.name, tool_call.arguments)

            # Execute the tool
            tool_result = self.tools.execute(tool_call.name, **tool_call.arguments)

            # Display the result
            self.display.print_tool_result(
                name=tool_call.name,
                result_content=tool_result.content,
                is_error=tool_result.is_error,
            )

            # Convert to message format for context
            from ..llm.messages import ToolResult as LLMToolResult
            llm_result = LLMToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=tool_result.content,
                is_error=tool_result.is_error,
            )
            result_messages.append(Message.tool_result(llm_result))

        return result_messages
