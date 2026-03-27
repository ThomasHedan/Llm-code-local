"""Grep content search tool for lclai."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult

MAX_RESULTS = 200
MAX_LINE_LENGTH = 300


class GrepSearchTool(BaseTool):
    """Tool to search file contents using regex patterns."""

    name = "grep_search"
    description = (
        "Search file contents for a regex pattern. "
        "Returns matching lines in 'file:line_number:content' format (max 200 results). "
        "Supports filtering by file glob pattern."
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
                            "description": "Regular expression pattern to search for.",
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "File or directory to search in. "
                                "Defaults to current working directory."
                            ),
                        },
                        "glob": {
                            "type": "string",
                            "description": (
                                "Glob pattern to filter files (e.g. '*.py', '**/*.ts'). "
                                "Only applies when path is a directory."
                            ),
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "If true, perform case-insensitive matching. Defaults to false.",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }

    def _search_with_ripgrep(
        self,
        pattern: str,
        search_path: Path,
        glob_pattern: str | None,
        case_insensitive: bool,
    ) -> tuple[list[str], bool]:
        """Try to search using ripgrep (rg) for speed.

        Returns:
            Tuple of (result_lines, success).
        """
        cmd = ["rg", "--line-number", "--no-heading", "--with-filename"]
        if case_insensitive:
            cmd.append("--ignore-case")
        if glob_pattern:
            cmd.extend(["--glob", glob_pattern])
        cmd.extend(["--", pattern, str(search_path)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode in (0, 1):  # 0=matches found, 1=no matches
                lines = [l for l in result.stdout.splitlines() if l]
                return lines, True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return [], False

    def _search_with_grep(
        self,
        pattern: str,
        search_path: Path,
        glob_pattern: str | None,
        case_insensitive: bool,
    ) -> tuple[list[str], bool]:
        """Try to search using system grep.

        Returns:
            Tuple of (result_lines, success).
        """
        cmd = ["grep", "-rn", "--with-filename"]
        if case_insensitive:
            cmd.append("-i")
        if glob_pattern and search_path.is_dir():
            cmd.extend(["--include", glob_pattern])
        cmd.extend(["-E", "--", pattern, str(search_path)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode in (0, 1):
                lines = [l for l in result.stdout.splitlines() if l]
                return lines, True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return [], False

    def _search_python(
        self,
        pattern: str,
        search_path: Path,
        glob_pattern: str | None,
        case_insensitive: bool,
    ) -> list[str]:
        """Pure Python fallback search using re module."""
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        results: list[str] = []

        if search_path.is_file():
            files_to_search = [search_path]
        else:
            # Collect files
            if glob_pattern:
                files_to_search = [f for f in search_path.glob(glob_pattern) if f.is_file()]
            else:
                files_to_search = [f for f in search_path.rglob("*") if f.is_file()]

        for file_path in sorted(files_to_search):
            # Skip binary files quickly
            try:
                raw = file_path.read_bytes()
                if b"\x00" in raw[:512]:
                    continue
                text = raw.decode("utf-8", errors="replace")
            except (PermissionError, OSError):
                continue

            for line_num, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    # Truncate very long lines
                    display_line = line if len(line) <= MAX_LINE_LENGTH else line[:MAX_LINE_LENGTH] + "..."
                    results.append(f"{file_path}:{line_num}:{display_line}")
                    if len(results) >= MAX_RESULTS:
                        return results

        return results

    def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_insensitive: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Search file contents for the given pattern.

        Args:
            pattern: Regex pattern to search for.
            path: File or directory to search in.
            glob: Optional file glob filter (e.g. '*.py').
            case_insensitive: Whether to ignore case.

        Returns:
            ToolResult with matching lines.
        """
        search_path = Path(path)

        if not search_path.exists():
            return ToolResult.error(f"Search path does not exist: {path}")

        result_lines: list[str] = []
        truncated = False

        # Try fast external tools first, then fall back to Python
        lines, ok = self._search_with_ripgrep(pattern, search_path, glob, case_insensitive)
        if not ok:
            lines, ok = self._search_with_grep(pattern, search_path, glob, case_insensitive)
        if not ok:
            try:
                lines = self._search_python(pattern, search_path, glob, case_insensitive)
            except ValueError as e:
                return ToolResult.error(str(e))

        if len(lines) > MAX_RESULTS:
            lines = lines[:MAX_RESULTS]
            truncated = True

        if not lines:
            return ToolResult.success(
                f"No matches found for pattern {pattern!r} in {path}"
            )

        output = "\n".join(lines)

        if truncated:
            output += (
                f"\n\n[Results truncated at {MAX_RESULTS} matches. "
                "Refine your pattern or use a more specific path/glob.]"
            )

        header = f"Found {len(lines)} match(es) for {pattern!r} in {path}:\n\n"
        return ToolResult.success(header + output)
