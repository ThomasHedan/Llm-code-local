"""Extract code examples from raw docstring text in multiple formats."""

from __future__ import annotations

import re

from .models import CodeExample


def extract_examples(docstring: str) -> list[CodeExample]:
    """Try all formats in order and merge results.

    Processes numpy, google, RST, markdown, and doctest formats.
    Deduplicates by code content.

    Args:
        docstring: Raw docstring text to extract examples from.

    Returns:
        List of CodeExample, doctest results first (most reliable).
    """
    if not docstring:
        return []

    seen_codes: set[str] = set()
    results: list[CodeExample] = []

    # Order: doctest first (most reliable), then structured formats
    for extractor in (
        _extract_doctest,
        _extract_numpy_examples,
        _extract_google_examples,
        _extract_rst_codeblocks,
        _extract_markdown_codeblocks,
    ):
        for example in extractor(docstring):
            normalized = example.code.strip()
            if normalized and normalized not in seen_codes:
                seen_codes.add(normalized)
                results.append(example)

    return results


def _extract_doctest(text: str) -> list[CodeExample]:
    """Find >>> lines anywhere in the docstring.

    Groups consecutive >>> lines (and their output lines) into one example.

    Args:
        text: Raw docstring text.

    Returns:
        List of CodeExample with code attribute set.

    Example:
        >>> x = [1, 2, 3]
        >>> sum(x)
        6
        → CodeExample(code="x = [1, 2, 3]\\nsum(x)")
    """
    examples: list[CodeExample] = []
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">>> "):
            current_lines.append(stripped[4:])  # Remove '>>> '
        elif stripped == ">>>":
            # Empty >>> line (blank continuation)
            current_lines.append("")
        elif stripped.startswith("... "):
            # Continuation line
            current_lines.append(stripped[4:])
        elif stripped == "...":
            # Empty continuation
            current_lines.append("")
        else:
            # Output line or blank — flush current block
            if current_lines:
                code = "\n".join(current_lines)
                if code.strip():
                    examples.append(CodeExample(code=code))
                current_lines = []

    if current_lines:
        code = "\n".join(current_lines)
        if code.strip():
            examples.append(CodeExample(code=code))

    return examples


def _extract_numpy_examples(text: str) -> list[CodeExample]:
    """Find 'Examples\\n--------' section and extract content until next section.

    Numpy-style docstrings use underlined section headers.

    Args:
        text: Raw docstring text.

    Returns:
        List of CodeExample extracted from the Examples section.
    """
    # Match 'Examples' section header (underlined with dashes or equals)
    section_header_re = re.compile(
        r"^[ \t]*(Examples)\s*\n[ \t]*[-=]+[ \t]*$",
        re.MULTILINE | re.IGNORECASE,
    )
    # Pattern to detect the start of any other section
    next_section_re = re.compile(
        r"^\s*\w[\w ]*\s*\n\s*[-=]+\s*$",
        re.MULTILINE,
    )

    match = section_header_re.search(text)
    if not match:
        return []

    section_start = match.end()

    # Find where next section starts (after the Examples section)
    remainder = text[section_start:]
    next_match = next_section_re.search(remainder)
    if next_match:
        section_text = remainder[: next_match.start()]
    else:
        section_text = remainder

    return _extract_doctest(section_text) or _extract_indented_code(section_text)


def _extract_google_examples(text: str) -> list[CodeExample]:
    """Find 'Examples:' followed by indented content.

    Google-style docstrings use 'Section:' headers.

    Args:
        text: Raw docstring text.

    Returns:
        List of CodeExample extracted from the Examples section.
    """
    # Match 'Examples:' header at start of a line (with optional leading spaces)
    header_re = re.compile(r"^[ \t]*Examples:\s*$", re.MULTILINE | re.IGNORECASE)

    match = header_re.search(text)
    if not match:
        return []

    section_start = match.end()
    remainder = text[section_start:]

    # Collect indented block — stop when we hit a non-indented non-blank line
    # that looks like another section header (e.g. "Returns:", "Args:")
    next_section_re = re.compile(r"^\S", re.MULTILINE)
    next_match = next_section_re.search(remainder)
    if next_match:
        section_text = remainder[: next_match.start()]
    else:
        section_text = remainder

    # Try doctest first, then indented code blocks
    results = _extract_doctest(section_text)
    if not results:
        results = _extract_indented_code(section_text)
    return results


def _extract_rst_codeblocks(text: str) -> list[CodeExample]:
    """Find '.. code-block:: python' or '.. code::' blocks.

    RST-style documentation uses directive syntax.

    Args:
        text: Raw docstring text.

    Returns:
        List of CodeExample from each code block found.
    """
    # Match RST code-block directive (with or without language specifier)
    directive_re = re.compile(
        r"^\s*\.\.\s+code(-block)?::[ \t]*(python|python3)?\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    examples: list[CodeExample] = []

    for m in directive_re.finditer(text):
        block_start = m.end()
        block_text = text[block_start:]
        indented = _collect_indented_block(block_text)
        if indented.strip():
            examples.append(CodeExample(code=indented.strip()))

    return examples


def _extract_markdown_codeblocks(text: str) -> list[CodeExample]:
    """Find ```python ... ``` blocks.

    Args:
        text: Raw docstring text.

    Returns:
        List of CodeExample from each fenced code block found.
    """
    # Match fenced code blocks with python language specifier
    fence_re = re.compile(
        r"```(?:python|python3)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    examples: list[CodeExample] = []
    for m in fence_re.finditer(text):
        code = m.group(1).strip()
        if code:
            examples.append(CodeExample(code=code))

    return examples


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_indented_code(text: str) -> list[CodeExample]:
    """Extract code from an indented block (fallback for section content).

    Args:
        text: Section text (already stripped of section header).

    Returns:
        List of CodeExample, one per contiguous indented block.
    """
    examples: list[CodeExample] = []
    current: list[str] = []

    for line in text.split("\n"):
        if line.strip() == "":
            if current:
                code = "\n".join(current).strip()
                if code:
                    examples.append(CodeExample(code=code))
                current = []
        elif line.startswith(("    ", "\t")):
            # Remove one level of indentation
            if line.startswith("    "):
                current.append(line[4:])
            else:
                current.append(line[1:])

    if current:
        code = "\n".join(current).strip()
        if code:
            examples.append(CodeExample(code=code))

    return examples


def _collect_indented_block(text: str) -> str:
    """Collect an indented block following a directive.

    Skips blank lines at the start, then collects lines until
    a non-indented, non-blank line is encountered.

    Args:
        text: Text starting right after a directive line.

    Returns:
        The collected block as a single string with indentation stripped.
    """
    lines = text.split("\n")
    result: list[str] = []
    started = False
    indent: str | None = None

    for line in lines:
        if not started:
            if line.strip() == "":
                continue  # skip leading blank lines
            # First non-blank line — determine indent level
            stripped_line = line.lstrip()
            indent = line[: len(line) - len(stripped_line)]
            if not indent:
                # No indentation → block ended before it started
                break
            started = True
            result.append(stripped_line)
        else:
            if line.strip() == "":
                result.append("")
            elif indent and line.startswith(indent):
                result.append(line[len(indent):])
            else:
                break

    return "\n".join(result)
