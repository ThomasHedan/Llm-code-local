"""File read tool for lclai."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult

MAX_LINES_DEFAULT = 300
LINE_NUMBER_WIDTH = 6  # Enough for 6-digit line numbers


class ReadFileTool(BaseTool):
    """Tool to read file contents with line numbers."""

    name = "read_file"
    description = (
        "Read the contents of a file with line numbers. "
        "Use offset and limit to read specific sections of large files."
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
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file to read.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": (
                                "Line number to start reading from (1-based). "
                                "Defaults to 1 (beginning of file)."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Maximum number of lines to read. "
                                f"Defaults to {MAX_LINES_DEFAULT}."
                            ),
                        },
                    },
                    "required": ["path"],
                },
            },
        }

    def execute(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Read a file and return its contents with line numbers.

        Args:
            path: Path to the file.
            offset: Starting line (1-based). Defaults to 1.
            limit: Max lines to return. Defaults to MAX_LINES_DEFAULT.

        Returns:
            ToolResult with numbered file content.
        """
        file_path = Path(path)

        if not file_path.exists():
            return ToolResult.error(f"File not found: {path}")

        if not file_path.is_file():
            return ToolResult.error(f"Path is not a file: {path}")

        try:
            # Detect binary files
            try:
                content_bytes = file_path.read_bytes()
                # Check for null bytes — a reasonable binary indicator
                if b"\x00" in content_bytes[:8192]:
                    return ToolResult.error(
                        f"File appears to be binary and cannot be displayed as text: {path}"
                    )
                text = content_bytes.decode("utf-8", errors="replace")
            except PermissionError:
                return ToolResult.error(f"Permission denied reading file: {path}")

            lines = text.splitlines(keepends=True)
            total_lines = len(lines)

            # Determine slice parameters
            start_line = max(1, offset or 1)
            max_lines = limit if limit is not None else MAX_LINES_DEFAULT

            # Warn if file is large and no offset/limit specified
            warning = ""
            if total_lines > MAX_LINES_DEFAULT and offset is None and limit is None:
                warning = (
                    f"[Warning: File has {total_lines} lines. "
                    f"Showing first {MAX_LINES_DEFAULT} lines. "
                    f"Use offset/limit to read other sections.]\n\n"
                )

            # Convert to 0-based indexing for slicing
            start_idx = start_line - 1
            end_idx = start_idx + max_lines

            selected_lines = lines[start_idx:end_idx]

            if not selected_lines and total_lines > 0:
                return ToolResult.error(
                    f"Offset {start_line} is beyond end of file ({total_lines} lines): {path}"
                )

            # Format with line numbers (cat -n style)
            numbered_lines = []
            for i, line in enumerate(selected_lines, start=start_line):
                # Remove trailing newline for display, we'll join with \n
                line_content = line.rstrip("\n").rstrip("\r")
                numbered_lines.append(f"{i:{LINE_NUMBER_WIDTH}d}\t{line_content}")

            result = "\n".join(numbered_lines)

            # Add trailing info
            end_line = start_line + len(selected_lines) - 1
            footer = ""
            if end_line < total_lines:
                remaining = total_lines - end_line
                footer = (
                    f"\n\n[Showing lines {start_line}-{end_line} of {total_lines}. "
                    f"{remaining} more lines not shown.]"
                )

            return ToolResult.success(f"{warning}{result}{footer}")

        except OSError as e:
            return ToolResult.error(f"Error reading file {path}: {e}")
