"""Data models for the doc_reader module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeExample:
    """A single runnable code example extracted from a docstring."""

    code: str              # Actual runnable code
    description: str = ""  # What this example demonstrates
    source: str = ""       # "module.Class.method" or "file:line"


@dataclass
class DocEntry:
    """Documentation entry for a function, method, or class."""

    name: str              # function or class name (short)
    qualified_name: str    # module.Class.method
    module: str            # e.g. "pandas.core.frame"
    signature: str         # "def groupby(self, by, ...)"
    summary: str           # First line of docstring
    examples: list[CodeExample] = field(default_factory=list)
    source_file: str = ""
    source_line: int = 0
