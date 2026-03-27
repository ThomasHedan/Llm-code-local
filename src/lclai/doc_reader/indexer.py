"""In-memory keyword-based index for DocEntry objects."""

from __future__ import annotations

import re
from collections import defaultdict

from .models import CodeExample, DocEntry


class DocIndex:
    """In-memory keyword index for DocEntry objects.

    Uses keyword-based search (no embeddings) for lightweight operation.
    Thread-safety is not guaranteed; use a single thread or add locking.
    """

    def __init__(self) -> None:
        self._entries: dict[str, DocEntry] = {}           # qualified_name → DocEntry
        self._keyword_index: dict[str, set[str]] = defaultdict(set)  # word → qualified_names
        self._packages: set[str] = set()

    def add(self, entry: DocEntry) -> None:
        """Add a DocEntry to the index.

        Indexes keywords from the entry name, summary, and example code.

        Args:
            entry: DocEntry to add.
        """
        self._entries[entry.qualified_name] = entry

        # Index name and summary words
        for word in self._tokenize(entry.name + " " + entry.summary):
            self._keyword_index[word].add(entry.qualified_name)

        # Index words from example code (first 20 tokens only)
        for ex in entry.examples:
            for word in self._tokenize(ex.code)[:20]:
                self._keyword_index[word].add(entry.qualified_name)

        # Track top-level package
        top_pkg = entry.module.split(".")[0]
        if top_pkg:
            self._packages.add(top_pkg)

    def add_many(self, entries: list[DocEntry]) -> None:
        """Add multiple DocEntry objects to the index.

        Args:
            entries: List of DocEntry objects to add.
        """
        for e in entries:
            self.add(e)
        if entries:
            top_pkg = entries[0].module.split(".")[0]
            if top_pkg:
                self._packages.add(top_pkg)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_examples: int = 1,
    ) -> list[DocEntry]:
        """Keyword search returning the top-k most relevant entries.

        Scoring:
        - 3 pts: query word appears in entry.name
        - 2 pts: query word appears in entry.qualified_name
        - 1 pt: query word appears in entry.summary or example code

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            min_examples: Only return entries with at least this many examples.

        Returns:
            Sorted list of DocEntry (highest score first), up to top_k entries.
        """
        query_words = self._tokenize(query)
        scores: dict[str, int] = defaultdict(int)

        for word in query_words:
            for qname in self._keyword_index.get(word, set()):
                entry = self._entries[qname]
                if entry.name.lower() == word:
                    scores[qname] += 3
                elif word in entry.qualified_name.lower():
                    scores[qname] += 2
                else:
                    scores[qname] += 1

        results = [
            (score, self._entries[qname])
            for qname, score in scores.items()
            if len(self._entries[qname].examples) >= min_examples
        ]
        results.sort(key=lambda x: -x[0])
        return [e for _, e in results[:top_k]]

    def get(self, qualified_name: str) -> DocEntry | None:
        """Retrieve a single DocEntry by its qualified name.

        Args:
            qualified_name: Fully qualified name (e.g. "pandas.core.frame.DataFrame.groupby").

        Returns:
            The DocEntry, or None if not found.
        """
        return self._entries.get(qualified_name)

    def stats(self) -> dict[str, object]:
        """Return statistics about the current index state.

        Returns:
            Dict with counts and indexed package names.
        """
        total_examples = sum(len(e.examples) for e in self._entries.values())
        return {
            "total_entries": len(self._entries),
            "entries_with_examples": sum(
                1 for e in self._entries.values() if e.examples
            ),
            "total_examples": total_examples,
            "indexed_packages": sorted(self._packages),
        }

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric characters, filter short words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens (length > 2).
        """
        return [
            w for w in re.split(r"[^a-z0-9_]", text.lower()) if len(w) > 2
        ]

    @property
    def is_empty(self) -> bool:
        """True if no entries have been indexed."""
        return len(self._entries) == 0
