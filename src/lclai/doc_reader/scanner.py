"""Scan files, directories, and installed packages for DocEntry objects."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from .models import DocEntry
from .parser import parse_file

# Default directories to exclude when scanning
_DEFAULT_EXCLUDES = frozenset({
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "dist",
    "build",
    ".eggs",
    ".tox",
    ".venv",
    "venv",
    "env",
    ".env",
    "migrations",
})


def scan_file(filepath: str) -> list[DocEntry]:
    """Parse a single Python file.

    Args:
        filepath: Path to the .py file to parse.

    Returns:
        List of DocEntry objects found in the file.
    """
    return parse_file(filepath)


def scan_directory(
    dirpath: str,
    pattern: str = "**/*.py",
    exclude: list[str] | None = None,
) -> list[DocEntry]:
    """Recursively scan a directory for .py files matching a pattern.

    Args:
        dirpath: Root directory to scan.
        pattern: Glob pattern for file matching (default "**/*.py").
        exclude: Additional directory names to exclude.

    Returns:
        All DocEntry objects found across matching files.
    """
    root = Path(dirpath).resolve()
    if not root.is_dir():
        return []

    excluded = _DEFAULT_EXCLUDES.copy()
    if exclude:
        excluded.update(exclude)

    entries: list[DocEntry] = []

    for py_file in root.glob(pattern):
        # Only examine the path relative to the scan root
        rel = py_file.relative_to(root)
        rel_parts = rel.parts  # e.g. ("subpkg", "module.py")

        # Skip files in excluded directories (check all directory components)
        dir_parts = set(rel_parts[:-1])
        if dir_parts & excluded:
            continue

        # Also skip test directories/files within the scanned tree
        skip = False
        for part in rel_parts:
            if part in excluded:
                skip = True
                break
            # Skip test dirs/files (but only within the scanned tree)
            if part.startswith("test") or part.startswith("_test"):
                skip = True
                break
        if skip:
            continue

        file_entries = parse_file(py_file)
        entries.extend(file_entries)

    return entries


def scan_installed_package(package_name: str) -> list[DocEntry]:
    """Scan an installed Python package for DocEntry objects.

    Finds the installed package location via importlib, then scans
    the package directory.

    Args:
        package_name: Top-level package name (e.g. "requests", "pathlib").

    Returns:
        All DocEntry objects found in the package.

    Raises:
        ValueError: If the package cannot be found or imported.
    """
    pkg_root = find_package_root(package_name)
    if pkg_root is None:
        raise ValueError(
            f"Package {package_name!r} not found. "
            "Make sure it is installed in the current environment."
        )

    if pkg_root.is_dir():
        return scan_directory(str(pkg_root))
    elif pkg_root.is_file():
        return scan_file(str(pkg_root))
    else:
        return []


def find_package_root(package_name: str) -> Path | None:
    """Find the root directory of an installed package.

    Tries to import the package and inspect its __path__ or __file__.

    Args:
        package_name: Package name to look up.

    Returns:
        Path to the package root directory, or None if not found.
    """
    try:
        mod = import_module(package_name)
        if hasattr(mod, "__path__"):
            path_list = list(mod.__path__)
            if path_list:
                return Path(path_list[0])
        if hasattr(mod, "__file__") and mod.__file__:
            return Path(mod.__file__).parent
    except ImportError:
        pass

    # Fallback: search sys.path manually
    for sys_path in sys.path:
        candidate = Path(sys_path) / package_name
        if candidate.is_dir() and (candidate / "__init__.py").exists():
            return candidate
        # Single-file module
        candidate_file = Path(sys_path) / f"{package_name}.py"
        if candidate_file.is_file():
            return candidate_file

    return None
