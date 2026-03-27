"""File edit tool for lclai — precise string replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult


class EditFileTool(BaseTool):
    """Tool to make precise edits to files via string replacement."""

    name = "edit_file"
    description = (
        "Edit a file by replacing a specific string with a new string. "
        "The file must exist and be readable. "
        "old_string must appear exactly once in the file (unless replace_all=True). "
        "Always read the file first to get the exact text to replace."
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
                            "description": "Absolute or relative path to the file to edit.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": (
                                "The exact string to find and replace. "
                                "Must be unique in the file when replace_all is false."
                            ),
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The string to replace old_string with.",
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": (
                                "If true, replace all occurrences of old_string. "
                                "Defaults to false."
                            ),
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        }

    def execute(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Replace old_string with new_string in the given file.

        Args:
            path: Path to the file to edit.
            old_string: String to find.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences.

        Returns:
            ToolResult with a diff summary or error.
        """
        file_path = Path(path)

        if not file_path.exists():
            return ToolResult.error(
                f"File not found: {path}. "
                "Use write_file to create a new file."
            )

        if not file_path.is_file():
            return ToolResult.error(f"Path is not a file: {path}")

        try:
            original = file_path.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return ToolResult.error(f"Permission denied reading file: {path}")
        except OSError as e:
            return ToolResult.error(f"Error reading file {path}: {e}")

        # Count occurrences
        count = original.count(old_string)

        if count == 0:
            # Provide helpful context about what IS in the file
            preview = original[:500] + ("..." if len(original) > 500 else "")
            return ToolResult.error(
                f"old_string not found in {path}.\n"
                f"The string to find was:\n{old_string!r}\n\n"
                f"File begins with:\n{preview}"
            )

        if count > 1 and not replace_all:
            # Show line numbers where it appears
            lines = original.splitlines()
            occurrences = [
                i + 1
                for i, line in enumerate(lines)
                if old_string in line
            ]
            # Also check multi-line occurrences
            occ_list = ", ".join(str(n) for n in occurrences[:10])
            return ToolResult.error(
                f"old_string appears {count} times in {path} "
                f"(near lines: {occ_list}). "
                "Provide more context to make it unique, "
                "or set replace_all=true to replace all occurrences."
            )

        try:
            if replace_all:
                new_content = original.replace(old_string, new_string)
            else:
                new_content = original.replace(old_string, new_string, 1)

            file_path.write_text(new_content, encoding="utf-8")
        except PermissionError:
            return ToolResult.error(f"Permission denied writing to: {path}")
        except OSError as e:
            return ToolResult.error(f"Error writing file {path}: {e}")

        # Build a diff summary
        old_lines = old_string.splitlines()
        new_lines = new_string.splitlines()
        replaced_count = count if replace_all else 1

        diff_lines = []
        for line in old_lines[:5]:
            diff_lines.append(f"  - {line}")
        if len(old_lines) > 5:
            diff_lines.append(f"  - ... ({len(old_lines) - 5} more lines)")
        for line in new_lines[:5]:
            diff_lines.append(f"  + {line}")
        if len(new_lines) > 5:
            diff_lines.append(f"  + ... ({len(new_lines) - 5} more lines)")

        diff_str = "\n".join(diff_lines)
        return ToolResult.success(
            f"Edited {path}: replaced {replaced_count} occurrence(s).\n{diff_str}"
        )
