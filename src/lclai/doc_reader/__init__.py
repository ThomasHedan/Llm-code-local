"""Doc reader module — parse, index, and query Python docstring examples."""

from .indexer import DocIndex
from .models import CodeExample, DocEntry
from .query import DocQuery
from .scanner import (
    find_package_root,
    scan_directory,
    scan_file,
    scan_installed_package,
)

__all__ = [
    "CodeExample",
    "DocEntry",
    "DocIndex",
    "DocQuery",
    "find_package_root",
    "scan_directory",
    "scan_file",
    "scan_installed_package",
]
