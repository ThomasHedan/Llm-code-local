"""AST-based Python file parser → list[DocEntry]."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from .formats import extract_examples
from .models import CodeExample, DocEntry


def parse_file(filepath: str | Path) -> list[DocEntry]:
    """Parse a Python file and return DocEntry for every function/class/method with a docstring.

    Args:
        filepath: Path to the .py file to parse.

    Returns:
        List of DocEntry objects found in the file.
    """
    filepath = Path(filepath)
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    module_name = _module_name_from_path(filepath)
    visitor = _DocVisitor(module_name=module_name, source_file=str(filepath))
    visitor.visit(tree)
    return visitor.entries


class _DocVisitor(ast.NodeVisitor):
    """AST visitor that collects DocEntry objects from function/class nodes."""

    def __init__(self, module_name: str, source_file: str) -> None:
        self.entries: list[DocEntry] = []
        self._module = module_name
        self._file = source_file
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        docstring = ast.get_docstring(node)
        if docstring:
            entry = self._make_entry(node, docstring, is_class=True)
            self.entries.append(entry)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        docstring = ast.get_docstring(node)
        if docstring:
            entry = self._make_entry(node, docstring, is_class=False)
            self.entries.append(entry)
        # Do not recurse into nested functions to avoid noise

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _make_entry(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        docstring: str,
        is_class: bool,
    ) -> DocEntry:
        """Build a DocEntry from an AST node.

        Args:
            node: The AST node (ClassDef or FunctionDef/AsyncFunctionDef).
            docstring: Extracted docstring text.
            is_class: True if the node is a class definition.

        Returns:
            DocEntry populated from the node.
        """
        # Build qualified name: module.ClassName.method_name
        parts = [self._module] + self._class_stack + [node.name]
        qualified_name = ".".join(p for p in parts if p)

        # Signature
        if is_class:
            signature = f"class {node.name}"
            # Include base classes if present
            if isinstance(node, ast.ClassDef) and node.bases:
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except Exception:
                        pass
                if bases:
                    signature += f"({', '.join(bases)})"
        else:
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            signature = _build_signature(node)

        # Summary: first non-empty line of docstring
        summary = ""
        for line in docstring.split("\n"):
            stripped = line.strip()
            if stripped:
                summary = stripped
                break

        # Examples
        examples = extract_examples(docstring)

        return DocEntry(
            name=node.name,
            qualified_name=qualified_name,
            module=self._module,
            signature=signature,
            summary=summary,
            examples=examples,
            source_file=self._file,
            source_line=node.lineno,
        )


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct def signature string from AST node.

    Handles positional args, *args, **kwargs, defaults, keyword-only args,
    and return annotations.

    Args:
        node: Function definition AST node.

    Returns:
        Signature string like "def func(self, x: int, y: str = 'hello') -> bool".
    """
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = node.args

    # Collect all argument representations
    arg_strings: list[str] = []
    all_args = args.args
    # Number of args with defaults (defaults are right-aligned)
    defaults_offset = len(all_args) - len(args.defaults)

    for i, arg in enumerate(all_args):
        arg_str = arg.arg
        if arg.annotation is not None:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        default_idx = i - defaults_offset
        if default_idx >= 0:
            try:
                arg_str += f" = {ast.unparse(args.defaults[default_idx])}"
            except Exception:
                pass
        arg_strings.append(arg_str)

    # *args
    if args.vararg:
        va = args.vararg
        va_str = f"*{va.arg}"
        if va.annotation is not None:
            try:
                va_str += f": {ast.unparse(va.annotation)}"
            except Exception:
                pass
        arg_strings.append(va_str)
    elif args.kwonlyargs:
        # Bare * separator when there's no *args but there are keyword-only args
        arg_strings.append("*")

    # Keyword-only args
    kw_defaults = args.kw_defaults
    for i, kwarg in enumerate(args.kwonlyargs):
        kw_str = kwarg.arg
        if kwarg.annotation is not None:
            try:
                kw_str += f": {ast.unparse(kwarg.annotation)}"
            except Exception:
                pass
        if i < len(kw_defaults) and kw_defaults[i] is not None:
            try:
                kw_str += f" = {ast.unparse(kw_defaults[i])}"  # type: ignore[arg-type]
            except Exception:
                pass
        arg_strings.append(kw_str)

    # **kwargs
    if args.kwarg:
        kw = args.kwarg
        kw_str = f"**{kw.arg}"
        if kw.annotation is not None:
            try:
                kw_str += f": {ast.unparse(kw.annotation)}"
            except Exception:
                pass
        arg_strings.append(kw_str)

    params = ", ".join(arg_strings)
    sig = f"{prefix} {node.name}({params})"

    # Return annotation
    if node.returns is not None:
        try:
            sig += f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass

    return sig


def _module_name_from_path(filepath: Path, base: Path | None = None) -> str:
    """Convert a filesystem path to a dotted module name.

    Walks up the directory tree looking for the last directory containing
    an __init__.py to find the package root.

    Args:
        filepath: Path to the .py file.
        base: Optional base directory to use as root.

    Returns:
        Dotted module name like "pandas.core.frame", or stem if not in a package.
    """
    filepath = filepath.resolve()

    if base is not None:
        base = base.resolve()
        try:
            rel = filepath.relative_to(base)
            parts = list(rel.parts)
            if parts[-1].endswith(".py"):
                parts[-1] = parts[-1][:-3]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)
        except ValueError:
            pass

    # Walk up to find the top-level package root (last dir with __init__.py)
    parts: list[str] = []
    current = filepath

    # Start from the file itself
    stem = current.stem
    if stem != "__init__":
        parts.append(stem)
    current = current.parent

    # Walk up while there's an __init__.py
    while (current / "__init__.py").exists():
        parts.append(current.name)
        current = current.parent

    parts.reverse()
    return ".".join(parts) if parts else filepath.stem
