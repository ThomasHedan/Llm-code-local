"""High-level query interface for the documentation index."""

from __future__ import annotations

from .indexer import DocIndex
from .models import DocEntry


class DocQuery:
    """Query interface that formats documentation examples for LLM injection."""

    def __init__(self, index: DocIndex) -> None:
        """Initialize DocQuery with a DocIndex.

        Args:
            index: The documentation index to query.
        """
        self._index = index

    def find_examples(
        self,
        query: str,
        max_tokens: int = 800,
        top_k: int = 3,
    ) -> str:
        """Search the index and format results as a compact string for LLM injection.

        Respects a token budget (approximated as 4 chars per token).
        Returns an empty string if the index is empty or no results are found.

        Args:
            query: Search query string.
            max_tokens: Maximum approximate token budget for the output.
            top_k: Maximum number of functions to return examples for.

        Returns:
            Formatted markdown string with code examples, or empty string.
        """
        if self._index.is_empty:
            return ""

        entries = self._index.search(query, top_k=top_k, min_examples=1)
        if not entries:
            return ""

        budget_chars = max_tokens * 4
        parts: list[str] = []
        used = 0

        for entry in entries:
            block = self._format_entry(entry)
            if used + len(block) > budget_chars:
                # Try to fit at least one example from this entry
                for ex in entry.examples[:1]:
                    mini = (
                        f"# {entry.qualified_name}\n"
                        f"```python\n{ex.code}\n```\n"
                    )
                    if used + len(mini) <= budget_chars:
                        parts.append(mini)
                        used += len(mini)
                break
            parts.append(block)
            used += len(block)

        if not parts:
            return ""

        header = "# Relevant examples from documentation (indexed)\n\n"
        return header + "\n".join(parts)

    def _format_entry(self, entry: DocEntry) -> str:
        """Format a single DocEntry as a markdown block.

        Args:
            entry: DocEntry to format.

        Returns:
            Formatted markdown string.
        """
        lines: list[str] = [f"## `{entry.qualified_name}`"]

        if entry.signature:
            lines.append(f"```python\n{entry.signature}\n```")

        if entry.summary:
            lines.append(entry.summary)

        if entry.examples:
            lines.append("**Examples:**")
            for ex in entry.examples[:3]:  # max 3 examples per entry
                if ex.description:
                    lines.append(f"_{ex.description}_")
                lines.append(f"```python\n{ex.code.strip()}\n```")

        return "\n".join(lines) + "\n"
