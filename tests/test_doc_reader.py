"""Comprehensive tests for the doc_reader module (Phase 2)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lclai.doc_reader.formats import (
    _extract_doctest,
    _extract_google_examples,
    _extract_markdown_codeblocks,
    _extract_numpy_examples,
    _extract_rst_codeblocks,
    extract_examples,
)
from lclai.doc_reader.indexer import DocIndex
from lclai.doc_reader.models import CodeExample, DocEntry
from lclai.doc_reader.parser import _build_signature, _module_name_from_path, parse_file
from lclai.doc_reader.query import DocQuery
from lclai.doc_reader.scanner import (
    find_package_root,
    scan_directory,
    scan_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    name: str = "my_func",
    qualified_name: str = "mymod.my_func",
    module: str = "mymod",
    summary: str = "Does something.",
    examples: list[CodeExample] | None = None,
) -> DocEntry:
    return DocEntry(
        name=name,
        qualified_name=qualified_name,
        module=module,
        signature=f"def {name}()",
        summary=summary,
        examples=examples or [],
    )


# ===========================================================================
# TestCodeExampleExtraction
# ===========================================================================


class TestCodeExampleExtraction:
    def test_extract_doctest_simple(self) -> None:
        docstring = ">>> x = 1\n>>> x\n1"
        examples = _extract_doctest(docstring)
        assert len(examples) == 1
        assert examples[0].code == "x = 1\nx"

    def test_extract_doctest_multiline(self) -> None:
        docstring = textwrap.dedent("""\
            >>> def add(a, b):
            ...     return a + b
            >>> add(1, 2)
            3
        """)
        examples = _extract_doctest(docstring)
        assert len(examples) >= 1
        # First example contains the def lines
        first_code = examples[0].code
        assert "def add(a, b):" in first_code
        assert "return a + b" in first_code

    def test_extract_doctest_multiple_blocks(self) -> None:
        docstring = textwrap.dedent("""\
            >>> x = 10

            >>> y = x * 2
            >>> y
            20
        """)
        examples = _extract_doctest(docstring)
        # Two separate blocks (blank line separates them)
        assert len(examples) >= 1
        all_code = " ".join(e.code for e in examples)
        assert "x = 10" in all_code
        assert "y = x * 2" in all_code

    def test_extract_numpy_examples(self) -> None:
        docstring = textwrap.dedent("""\
            Summary line.

            Parameters
            ----------
            x : int
                A number.

            Examples
            --------
            >>> import numpy as np
            >>> np.array([1, 2, 3])
            array([1, 2, 3])
        """)
        examples = _extract_numpy_examples(docstring)
        assert len(examples) >= 1
        all_code = "\n".join(e.code for e in examples)
        assert "import numpy as np" in all_code or "np.array" in all_code

    def test_extract_numpy_examples_no_section(self) -> None:
        docstring = "No examples here."
        examples = _extract_numpy_examples(docstring)
        assert examples == []

    def test_extract_google_examples(self) -> None:
        docstring = textwrap.dedent("""\
            Summary.

            Args:
                x: A value.

            Examples:
                >>> result = my_func(42)
                >>> result
                42
        """)
        examples = _extract_google_examples(docstring)
        assert len(examples) >= 1
        all_code = "\n".join(e.code for e in examples)
        assert "my_func(42)" in all_code

    def test_extract_google_examples_no_section(self) -> None:
        docstring = "Just a plain summary."
        examples = _extract_google_examples(docstring)
        assert examples == []

    def test_extract_rst_codeblock(self) -> None:
        docstring = textwrap.dedent("""\
            Summary.

            .. code-block:: python

                x = 1
                y = x + 1

            End of docs.
        """)
        examples = _extract_rst_codeblocks(docstring)
        assert len(examples) >= 1
        assert "x = 1" in examples[0].code
        assert "y = x + 1" in examples[0].code

    def test_extract_rst_codeblock_code_directive(self) -> None:
        docstring = textwrap.dedent("""\
            .. code:: python

                result = 42
        """)
        examples = _extract_rst_codeblocks(docstring)
        assert len(examples) >= 1
        assert "result = 42" in examples[0].code

    def test_extract_rst_no_codeblock(self) -> None:
        docstring = "No code blocks here."
        examples = _extract_rst_codeblocks(docstring)
        assert examples == []

    def test_extract_markdown_codeblock(self) -> None:
        docstring = textwrap.dedent("""\
            Summary.

            ```python
            x = [1, 2, 3]
            total = sum(x)
            ```
        """)
        examples = _extract_markdown_codeblocks(docstring)
        assert len(examples) == 1
        assert "x = [1, 2, 3]" in examples[0].code
        assert "total = sum(x)" in examples[0].code

    def test_extract_markdown_codeblock_multiple(self) -> None:
        docstring = textwrap.dedent("""\
            ```python
            a = 1
            ```

            Some text.

            ```python
            b = 2
            ```
        """)
        examples = _extract_markdown_codeblocks(docstring)
        assert len(examples) == 2

    def test_extract_markdown_no_codeblock(self) -> None:
        docstring = "No fences here."
        examples = _extract_markdown_codeblocks(docstring)
        assert examples == []

    def test_extract_empty_docstring(self) -> None:
        examples = extract_examples("")
        assert examples == []

    def test_extract_none_like_empty(self) -> None:
        examples = extract_examples("")
        assert isinstance(examples, list)
        assert len(examples) == 0

    def test_extract_no_examples(self) -> None:
        docstring = "This function does nothing special.\n\nArgs:\n    x: ignored."
        examples = extract_examples(docstring)
        assert examples == []

    def test_extract_deduplication(self) -> None:
        # Same code in both doctest AND a numpy section should appear once
        docstring = textwrap.dedent("""\
            >>> x = 1

            Examples
            --------
            >>> x = 1
        """)
        examples = extract_examples(docstring)
        codes = [e.code.strip() for e in examples]
        assert codes.count("x = 1") == 1

    def test_extract_returns_list(self) -> None:
        examples = extract_examples(">>> a = 1")
        assert isinstance(examples, list)
        for ex in examples:
            assert isinstance(ex, CodeExample)


# ===========================================================================
# TestDocParser
# ===========================================================================


class TestDocParser:
    def _write_temp_py(self, tmp_path: Path, content: str) -> Path:
        f = tmp_path / "test_module.py"
        f.write_text(textwrap.dedent(content))
        return f

    def test_parse_function_with_doctest(self, tmp_path: Path) -> None:
        code = '''\
            def add(a, b):
                """Add two numbers.

                >>> add(1, 2)
                3
                """
                return a + b
        '''
        filepath = self._write_temp_py(tmp_path, code)
        entries = parse_file(filepath)

        assert len(entries) == 1
        entry = entries[0]
        assert entry.name == "add"
        assert "Add two numbers" in entry.summary
        assert len(entry.examples) >= 1
        assert "add(1, 2)" in entries[0].examples[0].code

    def test_parse_class_with_examples(self, tmp_path: Path) -> None:
        code = '''\
            class Calculator:
                """A simple calculator.

                Examples:
                    >>> calc = Calculator()
                    >>> calc.add(1, 2)
                    3
                """

                def add(self, a, b):
                    """Add two values.

                    >>> c = Calculator()
                    >>> c.add(3, 4)
                    7
                    """
                    return a + b
        '''
        filepath = self._write_temp_py(tmp_path, code)
        entries = parse_file(filepath)

        names = [e.name for e in entries]
        assert "Calculator" in names
        assert "add" in names

        calc_entry = next(e for e in entries if e.name == "Calculator")
        assert len(calc_entry.examples) >= 1

    def test_parse_no_docstring_skipped(self, tmp_path: Path) -> None:
        code = '''\
            def no_doc(x):
                return x * 2

            def with_doc(x):
                """Has a docstring."""
                return x
        '''
        filepath = self._write_temp_py(tmp_path, code)
        entries = parse_file(filepath)

        names = [e.name for e in entries]
        assert "no_doc" not in names
        assert "with_doc" in names

    def test_parse_signature_reconstruction(self, tmp_path: Path) -> None:
        code = '''\
            def greet(name: str, greeting: str = "Hello") -> str:
                """Greet someone.

                >>> greet("World")
                'Hello, World'
                """
                return f"{greeting}, {name}"
        '''
        filepath = self._write_temp_py(tmp_path, code)
        entries = parse_file(filepath)

        assert len(entries) == 1
        sig = entries[0].signature
        assert "greet" in sig
        assert "name" in sig
        assert "str" in sig

    def test_parse_returns_qualified_name(self, tmp_path: Path) -> None:
        code = '''\
            def compute(x):
                """Compute something.

                >>> compute(5)
                25
                """
                return x ** 2
        '''
        filepath = self._write_temp_py(tmp_path, code)
        entries = parse_file(filepath)

        assert len(entries) == 1
        qname = entries[0].qualified_name
        # Should contain the function name
        assert "compute" in qname

    def test_parse_async_function(self, tmp_path: Path) -> None:
        code = '''\
            async def fetch(url: str) -> str:
                """Fetch a URL.

                >>> import asyncio
                >>> asyncio.run(fetch("http://example.com"))
                '...'
                """
                return ""
        '''
        filepath = self._write_temp_py(tmp_path, code)
        entries = parse_file(filepath)

        assert len(entries) == 1
        assert "async def" in entries[0].signature

    def test_parse_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("def broken(:\n    pass")
        entries = parse_file(f)
        assert entries == []

    def test_parse_source_line_set(self, tmp_path: Path) -> None:
        code = '''\
            def first():
                """First function.

                >>> first()
                """
                pass

            def second():
                """Second function.

                >>> second()
                """
                pass
        '''
        filepath = self._write_temp_py(tmp_path, code)
        entries = parse_file(filepath)

        assert len(entries) == 2
        lines = [e.source_line for e in entries]
        assert lines[0] < lines[1]


# ===========================================================================
# TestDocIndex
# ===========================================================================


class TestDocIndex:
    def _make_entry_with_example(
        self,
        name: str,
        qname: str,
        module: str,
        summary: str,
        code: str,
    ) -> DocEntry:
        return DocEntry(
            name=name,
            qualified_name=qname,
            module=module,
            signature=f"def {name}()",
            summary=summary,
            examples=[CodeExample(code=code)],
        )

    def test_is_empty_initially(self) -> None:
        index = DocIndex()
        assert index.is_empty is True

    def test_not_empty_after_add(self) -> None:
        index = DocIndex()
        entry = make_entry(examples=[CodeExample(code="x = 1")])
        index.add(entry)
        assert index.is_empty is False

    def test_add_and_get(self) -> None:
        index = DocIndex()
        entry = make_entry(examples=[CodeExample(code="x = 1")])
        index.add(entry)

        retrieved = index.get(entry.qualified_name)
        assert retrieved is not None
        assert retrieved.name == entry.name

    def test_get_missing_returns_none(self) -> None:
        index = DocIndex()
        assert index.get("nonexistent.func") is None

    def test_search_by_name(self) -> None:
        index = DocIndex()
        entry = self._make_entry_with_example(
            "groupby", "pandas.DataFrame.groupby", "pandas", "Group data.", "df.groupby('x')"
        )
        index.add(entry)

        results = index.search("groupby")
        assert len(results) >= 1
        assert results[0].name == "groupby"

    def test_search_by_keyword(self) -> None:
        index = DocIndex()
        entry = self._make_entry_with_example(
            "read_csv", "pandas.read_csv", "pandas",
            "Read a CSV file into DataFrame.", "pd.read_csv('data.csv')"
        )
        index.add(entry)

        results = index.search("csv")
        assert any(e.name == "read_csv" for e in results)

    def test_search_returns_only_with_examples(self) -> None:
        index = DocIndex()
        # Entry with no examples
        no_ex = make_entry(name="plain", qualified_name="mod.plain", module="mod")
        # Entry with examples
        with_ex = make_entry(
            name="fancy",
            qualified_name="mod.fancy",
            module="mod",
            examples=[CodeExample(code="fancy()")],
        )
        index.add(no_ex)
        index.add(with_ex)

        results = index.search("plain fancy", min_examples=1)
        names = [e.name for e in results]
        assert "fancy" in names
        assert "plain" not in names

    def test_search_min_examples_zero(self) -> None:
        index = DocIndex()
        no_ex = make_entry(name="plain", qualified_name="mod.plain", module="mod", summary="plain func")
        index.add(no_ex)

        results = index.search("plain", min_examples=0)
        assert any(e.name == "plain" for e in results)

    def test_stats(self) -> None:
        index = DocIndex()
        e1 = make_entry(
            name="f1", qualified_name="m.f1", module="m",
            examples=[CodeExample(code="f1()"), CodeExample(code="f1(2)")]
        )
        e2 = make_entry(name="f2", qualified_name="m.f2", module="m")
        index.add(e1)
        index.add(e2)

        stats = index.stats()
        assert stats["total_entries"] == 2
        assert stats["entries_with_examples"] == 1
        assert stats["total_examples"] == 2

    def test_stats_indexed_packages(self) -> None:
        index = DocIndex()
        entry = self._make_entry_with_example(
            "get", "requests.api.get", "requests", "Send GET request.", "requests.get('url')"
        )
        index.add(entry)

        stats = index.stats()
        assert "requests" in stats["indexed_packages"]

    def test_add_many(self) -> None:
        index = DocIndex()
        entries = [
            make_entry(
                name=f"func{i}",
                qualified_name=f"mod.func{i}",
                module="mod",
                examples=[CodeExample(code=f"func{i}()")],
            )
            for i in range(5)
        ]
        index.add_many(entries)
        assert index.stats()["total_entries"] == 5

    def test_search_top_k(self) -> None:
        index = DocIndex()
        for i in range(10):
            entry = self._make_entry_with_example(
                f"func{i}", f"mod.func{i}", "mod", "common keyword here", f"func{i}()"
            )
            index.add(entry)

        results = index.search("common keyword", top_k=3)
        assert len(results) <= 3

    def test_search_empty_index_returns_empty(self) -> None:
        index = DocIndex()
        results = index.search("anything")
        assert results == []


# ===========================================================================
# TestDocQuery
# ===========================================================================


class TestDocQuery:
    def test_find_examples_empty_index(self) -> None:
        index = DocIndex()
        query = DocQuery(index)
        result = query.find_examples("pandas groupby")
        assert result == ""

    def test_find_examples_no_results(self) -> None:
        index = DocIndex()
        # Add an entry that won't match the query
        entry = make_entry(
            name="unrelated",
            qualified_name="mod.unrelated",
            module="mod",
            summary="Does unrelated things.",
            examples=[CodeExample(code="unrelated()")],
        )
        index.add(entry)

        query = DocQuery(index)
        result = query.find_examples("zzz_no_match_xyz")
        assert result == ""

    def test_find_examples_formats_correctly(self) -> None:
        index = DocIndex()
        entry = DocEntry(
            name="groupby",
            qualified_name="pandas.DataFrame.groupby",
            module="pandas",
            signature="def groupby(self, by)",
            summary="Group DataFrame rows by a key.",
            examples=[CodeExample(code="df.groupby('col').sum()")],
        )
        index.add(entry)

        query = DocQuery(index)
        result = query.find_examples("groupby")

        assert "pandas.DataFrame.groupby" in result
        assert "df.groupby('col').sum()" in result
        assert "```python" in result

    def test_find_examples_includes_header(self) -> None:
        index = DocIndex()
        entry = make_entry(
            name="func",
            qualified_name="mod.func",
            module="mod",
            examples=[CodeExample(code="func()")],
        )
        index.add(entry)

        query = DocQuery(index)
        result = query.find_examples("func")
        assert "Relevant examples from documentation" in result

    def test_find_examples_respects_token_budget(self) -> None:
        index = DocIndex()
        # Create entries with large code blocks
        for i in range(5):
            entry = DocEntry(
                name=f"func{i}",
                qualified_name=f"mod.func{i}",
                module="mod",
                signature=f"def func{i}()",
                summary="A function with large examples.",
                examples=[CodeExample(code="x = " + "a" * 500)],
            )
            index.add(entry)

        query = DocQuery(index)
        # Very small budget — should still return something but be capped
        result = query.find_examples("func", max_tokens=50, top_k=5)
        # Should not exceed budget significantly
        assert len(result) <= 50 * 4 * 3  # generous upper bound

    def test_find_examples_max_three_examples_per_entry(self) -> None:
        index = DocIndex()
        entry = DocEntry(
            name="func",
            qualified_name="mod.func",
            module="mod",
            signature="def func()",
            summary="A function.",
            examples=[
                CodeExample(code=f"example_{i}()") for i in range(6)
            ],
        )
        index.add(entry)

        query = DocQuery(index)
        result = query.find_examples("func", max_tokens=2000)
        # At most 3 examples should appear in formatted output
        count = result.count("```python")
        # +1 for signature block
        assert count <= 4  # 1 sig + 3 examples


# ===========================================================================
# TestScanner
# ===========================================================================


class TestScanner:
    def _write_temp_py(self, tmp_path: Path, name: str, content: str) -> Path:
        f = tmp_path / name
        f.write_text(textwrap.dedent(content))
        return f

    def test_scan_file(self, tmp_path: Path) -> None:
        code = '''\
            def documented():
                """A documented function.

                >>> documented()
                True
                """
                return True

            def undocumented():
                return False
        '''
        filepath = self._write_temp_py(tmp_path, "sample.py", code)
        entries = scan_file(str(filepath))

        names = [e.name for e in entries]
        assert "documented" in names
        assert "undocumented" not in names

    def test_scan_directory(self, tmp_path: Path) -> None:
        code_a = '''\
            def func_a():
                """Function A.

                >>> func_a()
                'a'
                """
                return "a"
        '''
        code_b = '''\
            def func_b():
                """Function B.

                >>> func_b()
                'b'
                """
                return "b"
        '''
        sub = tmp_path / "mypkg"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        self._write_temp_py(sub, "a.py", code_a)
        self._write_temp_py(sub, "b.py", code_b)

        entries = scan_directory(str(sub))
        names = [e.name for e in entries]
        assert "func_a" in names
        assert "func_b" in names

    def test_scan_directory_excludes_pycache(self, tmp_path: Path) -> None:
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        cached = pycache / "cached.py"
        cached.write_text('def cached():\n    """Cached.\n\n    >>> cached()\n    """\n    pass')

        entries = scan_directory(str(tmp_path))
        names = [e.name for e in entries]
        assert "cached" not in names

    def test_scan_directory_nonexistent(self, tmp_path: Path) -> None:
        entries = scan_directory(str(tmp_path / "does_not_exist"))
        assert entries == []

    def test_find_package_root_builtin(self) -> None:
        # 'json' is always available in the stdlib
        root = find_package_root("json")
        assert root is not None
        assert root.is_dir() or root.is_file()

    def test_find_package_root_pathlib(self) -> None:
        root = find_package_root("pathlib")
        # pathlib may be a single file module
        assert root is not None

    def test_find_package_root_not_found(self) -> None:
        root = find_package_root("_this_package_does_not_exist_xyz_abc_123")
        assert root is None

    def test_scan_file_string_path(self, tmp_path: Path) -> None:
        code = '''\
            def sample():
                """Sample function.

                >>> sample()
                1
                """
                return 1
        '''
        filepath = self._write_temp_py(tmp_path, "s.py", code)
        entries = scan_file(str(filepath))
        assert any(e.name == "sample" for e in entries)
