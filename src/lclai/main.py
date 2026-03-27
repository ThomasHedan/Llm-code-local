"""Entry point for lclai CLI."""

from __future__ import annotations

import sys

import click

from . import __version__


@click.command()
@click.option(
    "--model",
    "-m",
    default="qwen2.5-coder:7b",
    show_default=True,
    help="Ollama model to use.",
)
@click.option(
    "--base-url",
    default="http://localhost:11434",
    show_default=True,
    help="Ollama API base URL.",
)
@click.option(
    "--max-tokens",
    default=8000,
    show_default=True,
    type=int,
    help="Context window size in tokens.",
)
@click.version_option(version=__version__, prog_name="lclai")
def cli(model: str, base_url: str, max_tokens: int) -> None:
    """lclai — 100% local AI code assistant powered by Ollama.

    Start an interactive session to write, read, and understand code
    using a locally-running language model. No data leaves your machine.

    \b
    Examples:
      lclai
      lclai --model llama3.2:3b
      lclai --model codellama:13b --max-tokens 16000
      lclai --base-url http://192.168.1.10:11434

    \b
    Slash commands (once inside):
      /help     Show available commands
      /models   List available Ollama models
      /model    Switch model mid-session
      /clear    Clear conversation history
      /context  Show token usage
      /exit     Quit
    """
    # Defer heavy imports so --help and --version are instant
    from .llm.ollama import OllamaClient, OllamaError
    from .tools.registry import get_default_tools
    from .context.manager import ContextManager
    from .cli.display import Display
    from .cli.repl import REPL

    display = Display()

    # Initialize LLM client
    llm = OllamaClient(model=model, base_url=base_url)

    # Verify Ollama is reachable before entering the REPL
    try:
        llm._check_connection()
    except OllamaError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Initialize tools
    tools = get_default_tools()

    # Initialize context manager
    context = ContextManager(max_tokens=max_tokens)

    # Start the REPL
    repl = REPL(llm=llm, tools=tools, context=context, display=display)

    try:
        repl.run()
    except KeyboardInterrupt:
        display.console.print()
        display.print_info("Interrupted. Bye!")
        sys.exit(0)
