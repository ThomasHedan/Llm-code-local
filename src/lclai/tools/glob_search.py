"""Glob file search tool for lclai."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult

MAX_RESULTS = 100


class GlobSearchTool(BaseTool):
    """Tool to find files matching a glob pattern."""

    name = "glob_search"
    description = (
        "Find files matching a glob pattern. "
        "Supports patterns like '**/*.py', '*.toml', 'src/**/*.ts'. "
        "Returns sorted list of matching file paths (max 100 results)."
    )

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Glob pattern to match files against. "
                                "Examples: '**/*.py', 'src/**/*.ts', '*.json'"
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Directory to search in. "
                                "Defaults to current working directory."
                            ),
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }

    def execute(
        self,
        pattern: str,
        path: str = ".",
        **kwargs: Any,
    ) -> ToolResult:
        """Find files matching the given glob pattern.

        Args:
            pattern: Glob pattern to match.
            path: Root directory for the search. Defaults to '.'.

        Returns:
            ToolResult listing matching file paths.
        """
        search_root = Path(path)

        if not search_root.exists():
            return ToolResult.error(f"Search path does not exist: {path}")

        if not search_root.is_dir():
            return ToolResult.error(f"Search path is not a directory: {path}")

        try:
            # Use rglob for patterns starting with ** to avoid double-rooting issues
            if pattern.startswith("**/"):
                # Pattern already has **, use glob from root
                matches = list(search_root.glob(pattern))
            elif "**" in pattern:
                matches = list(search_root.glob(pattern))
            else:
                # Non-recursive pattern
                matches = list(search_root.glob(pattern))

            # Filter to only files (not directories)
            file_matches = [m for m in matches if m.is_file()]

            # Sort by path string for consistent output
            file_matches.sort(key=lambda p: str(p))

            truncated = False
            if len(file_matches) > MAX_RESULTS:
                file_matches = file_matches[:MAX_RESULTS]
                truncated = True

            if not file_matches:
                return ToolResult.success(
                    f"No files found matching pattern {pattern!r} in {path}"
                )

            lines = [str(p) for p in file_matches]
            output = "\n".join(lines)

            if truncated:
                output += (
                    f"\n\n[Results truncated at {MAX_RESULTS} files. "
                    "Refine your pattern to narrow results.]"
                )

            header = f"Found {len(file_matches)} file(s) matching {pattern!r} in {path}:\n\n"
            return ToolResult.success(header + output)

        except ValueError as e:
            return ToolResult.error(f"Invalid glob pattern {pattern!r}: {e}")
        except OSError as e:
            return ToolResult.error(f"Error searching in {path}: {e}")
