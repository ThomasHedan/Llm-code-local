"""File write tool for lclai."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult


class WriteFileTool(BaseTool):
    """Tool to write or create files."""

    name = "write_file"
    description = (
        "Write content to a file, creating it (and parent directories) if needed. "
        "This overwrites existing files completely. "
        "Prefer edit_file for making targeted changes to existing files."
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
                            "description": "Absolute or relative path to the file to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The full content to write to the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        }

    def execute(self, path: str, content: str, **kwargs: Any) -> ToolResult:
        """Write content to a file.

        Creates parent directories as needed. Overwrites existing files.

        Args:
            path: Path to write to.
            content: Content to write.

        Returns:
            ToolResult indicating success or error.
        """
        file_path = Path(path)

        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            existed = file_path.exists()
            file_path.write_text(content, encoding="utf-8")

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            if not content:
                line_count = 0

            action = "Updated" if existed else "Created"
            return ToolResult.success(
                f"{action} {path} ({line_count} lines, {len(content)} bytes)"
            )

        except PermissionError:
            return ToolResult.error(f"Permission denied writing to: {path}")
        except IsADirectoryError:
            return ToolResult.error(f"Path is a directory, not a file: {path}")
        except OSError as e:
            return ToolResult.error(f"Error writing file {path}: {e}")
