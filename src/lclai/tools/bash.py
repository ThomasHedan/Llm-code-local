"""Bash shell execution tool for lclai."""

from __future__ import annotations

import subprocess
from typing import Any

from .base import BaseTool, ToolResult

MAX_OUTPUT_CHARS = 5000

DANGEROUS_PATTERNS = [
    "rm -rf",
    "rm -fr",
    "> /dev/sda",
    "dd if=",
    "mkfs",
    ":(){ :|:& };:",  # fork bomb
    "chmod -R 777 /",
    "chown -R",
    "mv /* ",
    "wget -O- | bash",
    "curl | bash",
    "curl | sh",
]


class BashTool(BaseTool):
    """Tool to execute shell commands."""

    name = "bash"
    description = (
        "Execute a shell command and return its output (stdout + stderr combined). "
        "Use for running tests, installing packages, checking system state, etc. "
        "Commands run in a subprocess with a configurable timeout."
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
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": (
                                "Maximum seconds to wait for the command to complete. "
                                "Defaults to 30. Max 300."
                            ),
                        },
                    },
                    "required": ["command"],
                },
            },
        }

    def _check_dangerous(self, command: str) -> str | None:
        """Return a warning string if the command contains dangerous patterns."""
        lower_cmd = command.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in lower_cmd:
                return (
                    f"[Warning: Command contains potentially dangerous pattern: {pattern!r}. "
                    "Proceeding anyway — please verify the output carefully.]\n"
                )
        return None

    def execute(
        self,
        command: str,
        timeout: int = 30,
        **kwargs: Any,
    ) -> ToolResult:
        """Run a shell command and return combined stdout+stderr.

        Args:
            command: The shell command to execute.
            timeout: Maximum seconds to wait. Capped at 300.

        Returns:
            ToolResult with command output.
        """
        # Cap timeout at 5 minutes
        effective_timeout = min(int(timeout), 300)
        if effective_timeout < 1:
            effective_timeout = 30

        warning = self._check_dangerous(command)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            msg = f"Command timed out after {effective_timeout} seconds: {command!r}"
            if warning:
                msg = warning + msg
            return ToolResult.error(msg)
        except OSError as e:
            return ToolResult.error(f"Failed to run command: {e}")

        # Combine stdout and stderr
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            # Only add stderr header if both streams have content
            if result.stdout:
                output_parts.append("\n[stderr]\n" + result.stderr)
            else:
                output_parts.append(result.stderr)

        combined = "".join(output_parts)

        # Truncate if too long
        truncated = False
        if len(combined) > MAX_OUTPUT_CHARS:
            combined = combined[:MAX_OUTPUT_CHARS]
            truncated = True

        # Build final output
        lines = []
        if warning:
            lines.append(warning)

        if combined:
            lines.append(combined)
        else:
            lines.append("(no output)")

        if truncated:
            lines.append(
                f"\n[Output truncated at {MAX_OUTPUT_CHARS} characters. "
                "Use pipes or redirect to a file for full output.]"
            )

        if result.returncode != 0:
            lines.append(f"\n[Exit code: {result.returncode}]")

        output_text = "".join(lines)

        if result.returncode != 0:
            return ToolResult(content=output_text, is_error=True)
        return ToolResult.success(output_text)
