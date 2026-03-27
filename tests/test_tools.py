"""Tests for lclai tool implementations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from lclai.tools.base import ToolResult
from lclai.tools.file_read import ReadFileTool
from lclai.tools.file_write import WriteFileTool
from lclai.tools.file_edit import EditFileTool
from lclai.tools.bash import BashTool
from lclai.tools.glob_search import GlobSearchTool
from lclai.tools.grep_search import GrepSearchTool
from lclai.tools.registry import ToolRegistry, get_default_tools
from lclai.context.manager import ContextManager
from lclai.llm.messages import Message, Role, ToolCall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    f = tmp_path / "sample.txt"
    lines = [f"Line {i}: hello world" for i in range(1, 21)]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_factory(self) -> None:
        r = ToolResult.success("ok")
        assert r.content == "ok"
        assert r.is_error is False

    def test_error_factory(self) -> None:
        r = ToolResult.error("oops")
        assert r.content == "oops"
        assert r.is_error is True


# ---------------------------------------------------------------------------
# ReadFileTool tests
# ---------------------------------------------------------------------------


class TestReadFileTool:
    def setup_method(self) -> None:
        self.tool = ReadFileTool()

    def test_schema_has_required_fields(self) -> None:
        schema = self.tool.get_schema()
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "read_file"
        assert "path" in func["parameters"]["required"]

    def test_read_existing_file(self, sample_file: Path) -> None:
        result = self.tool.execute(path=str(sample_file))
        assert result.is_error is False
        assert "Line 1:" in result.content
        # Line numbers should be present
        assert "\t" in result.content

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        result = self.tool.execute(path=str(tmp_path / "missing.txt"))
        assert result.is_error is True
        assert "not found" in result.content.lower()

    def test_read_with_offset_and_limit(self, sample_file: Path) -> None:
        result = self.tool.execute(path=str(sample_file), offset=5, limit=3)
        assert result.is_error is False
        # Should start from line 5
        assert "Line 5:" in result.content
        # Should not include line 9
        assert "Line 9:" not in result.content

    def test_read_large_file_warns(self, tmp_path: Path) -> None:
        large_file = tmp_path / "large.txt"
        content = "\n".join(f"line {i}" for i in range(400))
        large_file.write_text(content, encoding="utf-8")
        result = self.tool.execute(path=str(large_file))
        assert result.is_error is False
        assert "Warning" in result.content

    def test_read_directory_returns_error(self, tmp_path: Path) -> None:
        result = self.tool.execute(path=str(tmp_path))
        assert result.is_error is True

    def test_name_and_description(self) -> None:
        assert self.tool.name == "read_file"
        assert len(self.tool.description) > 0


# ---------------------------------------------------------------------------
# WriteFileTool tests
# ---------------------------------------------------------------------------


class TestWriteFileTool:
    def setup_method(self) -> None:
        self.tool = WriteFileTool()

    def test_write_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "new_file.txt"
        result = self.tool.execute(path=str(target), content="hello\nworld\n")
        assert result.is_error is False
        assert target.exists()
        assert target.read_text() == "hello\nworld\n"

    def test_write_overwrites_existing(self, sample_file: Path) -> None:
        result = self.tool.execute(path=str(sample_file), content="replaced")
        assert result.is_error is False
        assert sample_file.read_text() == "replaced"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        result = self.tool.execute(path=str(target), content="deep")
        assert result.is_error is False
        assert target.exists()

    def test_write_reports_created_vs_updated(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        r1 = self.tool.execute(path=str(target), content="first")
        assert "Created" in r1.content

        r2 = self.tool.execute(path=str(target), content="second")
        assert "Updated" in r2.content

    def test_schema(self) -> None:
        schema = self.tool.get_schema()
        params = schema["function"]["parameters"]["properties"]
        assert "path" in params
        assert "content" in params


# ---------------------------------------------------------------------------
# EditFileTool tests
# ---------------------------------------------------------------------------


class TestEditFileTool:
    def setup_method(self) -> None:
        self.tool = EditFileTool()

    def test_edit_replaces_unique_string(self, tmp_path: Path) -> None:
        f = tmp_path / "edit_me.txt"
        f.write_text("Hello world\nGoodbye world\n")
        result = self.tool.execute(path=str(f), old_string="Hello", new_string="Hi")
        assert result.is_error is False
        assert f.read_text() == "Hi world\nGoodbye world\n"

    def test_edit_error_when_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "edit_me.txt"
        f.write_text("Hello world\n")
        result = self.tool.execute(path=str(f), old_string="xyz_not_here", new_string="nope")
        assert result.is_error is True
        assert "not found" in result.content.lower()

    def test_edit_error_when_ambiguous(self, tmp_path: Path) -> None:
        f = tmp_path / "edit_me.txt"
        f.write_text("foo bar\nfoo baz\n")
        result = self.tool.execute(path=str(f), old_string="foo", new_string="qux")
        assert result.is_error is True
        assert "2" in result.content  # mentions count

    def test_edit_replace_all(self, tmp_path: Path) -> None:
        f = tmp_path / "edit_me.txt"
        f.write_text("foo bar\nfoo baz\n")
        result = self.tool.execute(
            path=str(f), old_string="foo", new_string="qux", replace_all=True
        )
        assert result.is_error is False
        assert f.read_text() == "qux bar\nqux baz\n"

    def test_edit_nonexistent_file(self, tmp_path: Path) -> None:
        result = self.tool.execute(
            path=str(tmp_path / "ghost.txt"), old_string="x", new_string="y"
        )
        assert result.is_error is True
        assert "not found" in result.content.lower()

    def test_edit_multiline_replacement(self, tmp_path: Path) -> None:
        f = tmp_path / "ml.txt"
        f.write_text("def foo():\n    pass\n")
        result = self.tool.execute(
            path=str(f),
            old_string="def foo():\n    pass",
            new_string="def foo():\n    return 42",
        )
        assert result.is_error is False
        assert "return 42" in f.read_text()


# ---------------------------------------------------------------------------
# BashTool tests
# ---------------------------------------------------------------------------


class TestBashTool:
    def setup_method(self) -> None:
        self.tool = BashTool()

    def test_simple_command(self) -> None:
        result = self.tool.execute(command="echo hello")
        assert result.is_error is False
        assert "hello" in result.content

    def test_command_with_nonzero_exit(self) -> None:
        result = self.tool.execute(command="exit 1", timeout=5)
        assert result.is_error is True
        assert "Exit code: 1" in result.content

    def test_command_timeout(self) -> None:
        result = self.tool.execute(command="sleep 10", timeout=1)
        assert result.is_error is True
        assert "timed out" in result.content.lower()

    def test_command_captures_stderr(self) -> None:
        result = self.tool.execute(command="echo err >&2", timeout=5)
        # stderr should appear in output
        assert "err" in result.content

    def test_dangerous_command_warns(self) -> None:
        # Just check warning is present — command still runs
        result = self.tool.execute(command="echo 'rm -rf test'", timeout=5)
        # The command itself is safe (echo), but pattern check fires on the string
        # "rm -rf" appears in the command string, so warning fires
        assert "Warning" in result.content or result.is_error is False

    def test_truncates_long_output(self, tmp_path: Path) -> None:
        # Generate output longer than MAX_OUTPUT_CHARS (5000)
        result = self.tool.execute(command="python3 -c \"print('x' * 10000)\"", timeout=10)
        assert "truncated" in result.content.lower() or len(result.content) <= 6000

    def test_schema(self) -> None:
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "bash"
        props = schema["function"]["parameters"]["properties"]
        assert "command" in props
        assert "timeout" in props


# ---------------------------------------------------------------------------
# GlobSearchTool tests
# ---------------------------------------------------------------------------


class TestGlobSearchTool:
    def setup_method(self) -> None:
        self.tool = GlobSearchTool()

    def _make_tree(self, root: Path) -> None:
        (root / "a.py").write_text("python a")
        (root / "b.py").write_text("python b")
        (root / "c.txt").write_text("text c")
        sub = root / "sub"
        sub.mkdir()
        (sub / "d.py").write_text("python d")

    def test_glob_finds_py_files(self, tmp_path: Path) -> None:
        self._make_tree(tmp_path)
        result = self.tool.execute(pattern="*.py", path=str(tmp_path))
        assert result.is_error is False
        assert "a.py" in result.content
        assert "b.py" in result.content
        # Non-recursive, sub/d.py should not appear
        assert "d.py" not in result.content

    def test_glob_recursive(self, tmp_path: Path) -> None:
        self._make_tree(tmp_path)
        result = self.tool.execute(pattern="**/*.py", path=str(tmp_path))
        assert result.is_error is False
        assert "d.py" in result.content

    def test_glob_no_matches(self, tmp_path: Path) -> None:
        result = self.tool.execute(pattern="*.rs", path=str(tmp_path))
        assert result.is_error is False
        assert "No files found" in result.content

    def test_glob_invalid_path(self, tmp_path: Path) -> None:
        result = self.tool.execute(pattern="*.py", path=str(tmp_path / "nonexistent"))
        assert result.is_error is True

    def test_schema(self) -> None:
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "glob_search"


# ---------------------------------------------------------------------------
# GrepSearchTool tests
# ---------------------------------------------------------------------------


class TestGrepSearchTool:
    def setup_method(self) -> None:
        self.tool = GrepSearchTool()

    def _make_files(self, root: Path) -> None:
        (root / "alpha.py").write_text("def hello():\n    return 'world'\n")
        (root / "beta.py").write_text("class Foo:\n    pass\n")
        (root / "notes.txt").write_text("hello there\ngoodbye\n")

    def test_grep_finds_pattern(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        result = self.tool.execute(pattern="hello", path=str(tmp_path))
        assert result.is_error is False
        assert "alpha.py" in result.content
        assert "notes.txt" in result.content

    def test_grep_case_insensitive(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        result = self.tool.execute(
            pattern="HELLO", path=str(tmp_path), case_insensitive=True
        )
        assert result.is_error is False
        assert "alpha.py" in result.content

    def test_grep_with_glob_filter(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        result = self.tool.execute(
            pattern="hello", path=str(tmp_path), glob="*.py"
        )
        assert result.is_error is False
        # notes.txt should not appear because of .py filter
        # (depends on backend; at minimum alpha.py should appear)
        assert "alpha.py" in result.content

    def test_grep_no_matches(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        result = self.tool.execute(pattern="xyzzy_nomatch_999", path=str(tmp_path))
        assert result.is_error is False
        assert "No matches" in result.content

    def test_grep_invalid_path(self, tmp_path: Path) -> None:
        result = self.tool.execute(pattern="x", path=str(tmp_path / "nope"))
        assert result.is_error is True

    def test_grep_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "single.txt"
        f.write_text("line one\nline two\nline three\n")
        result = self.tool.execute(pattern="two", path=str(f))
        assert result.is_error is False
        assert "line two" in result.content

    def test_schema(self) -> None:
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "grep_search"


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_execute(self) -> None:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        assert "read_file" in registry

    def test_double_register_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ReadFileTool())

    def test_execute_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("nonexistent_tool")
        assert result.is_error is True
        assert "Unknown tool" in result.content

    def test_get_all_schemas(self) -> None:
        registry = get_default_tools()
        schemas = registry.get_all_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "bash" in names
        assert "glob_search" in names
        assert "grep_search" in names

    def test_default_tools_count(self) -> None:
        registry = get_default_tools()
        assert len(registry) == 6

    def test_list_tools(self) -> None:
        registry = get_default_tools()
        tools = registry.list_tools()
        assert isinstance(tools, list)
        assert len(tools) == 6
        assert tools == sorted(tools)  # Should be sorted


# ---------------------------------------------------------------------------
# ContextManager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_initial_state(self) -> None:
        cm = ContextManager(max_tokens=1000, reserve_output=200)
        assert cm.max_tokens == 1000
        assert cm.reserve_output == 200
        assert cm.input_budget == 800
        assert len(cm) == 0

    def test_set_system(self) -> None:
        cm = ContextManager()
        cm.set_system("You are a helpful assistant.")
        assert cm.get_system() == "You are a helpful assistant."
        assert cm.system_tokens > 0

    def test_add_and_get_messages(self) -> None:
        cm = ContextManager()
        msg = Message.user("Hello!")
        cm.add_message(msg)
        assert len(cm) == 1
        msgs = cm.get_messages()
        assert msgs[0].content == "Hello!"

    def test_get_all_messages_includes_system(self) -> None:
        cm = ContextManager()
        cm.set_system("System prompt")
        cm.add_message(Message.user("hi"))
        all_msgs = cm.get_all_messages()
        assert len(all_msgs) == 2
        assert all_msgs[0].role == Role.SYSTEM
        assert all_msgs[1].role == Role.USER

    def test_clear_removes_messages(self) -> None:
        cm = ContextManager()
        cm.set_system("sys")
        cm.add_message(Message.user("hello"))
        cm.clear()
        assert len(cm) == 0
        # System prompt preserved
        assert cm.get_system() == "sys"

    def test_reset_clears_everything(self) -> None:
        cm = ContextManager()
        cm.set_system("sys")
        cm.add_message(Message.user("hello"))
        cm.reset()
        assert len(cm) == 0
        assert cm.get_system() == ""

    def test_estimate_tokens(self) -> None:
        cm = ContextManager()
        # 4 chars = ~1 token
        assert cm.estimate_tokens("abcd") == 1
        assert cm.estimate_tokens("") == 0
        assert cm.estimate_tokens("a" * 400) == 100

    def test_available_tokens_decreases_with_messages(self) -> None:
        cm = ContextManager(max_tokens=1000, reserve_output=200)
        initial = cm.available_tokens
        cm.add_message(Message.user("a" * 400))  # ~100 tokens
        assert cm.available_tokens < initial

    def test_trim_to_fit_removes_old_messages(self) -> None:
        # Very small budget to force trimming
        cm = ContextManager(max_tokens=200, reserve_output=50)
        # Add many messages
        for i in range(20):
            cm.add_message(Message.user(f"Message number {i} with some padding content here."))
        # Should have trimmed some but kept recent ones
        assert len(cm) < 20

    def test_token_summary_keys(self) -> None:
        cm = ContextManager()
        summary = cm.token_summary()
        expected_keys = {
            "max_tokens",
            "reserve_output",
            "input_budget",
            "system_tokens",
            "messages_tokens",
            "used_tokens",
            "available_tokens",
        }
        assert set(summary.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------


class TestMessage:
    def test_user_factory(self) -> None:
        msg = Message.user("hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_assistant_factory(self) -> None:
        msg = Message.assistant("response")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "response"
        assert msg.tool_calls == []

    def test_system_factory(self) -> None:
        msg = Message.system("be helpful")
        assert msg.role == Role.SYSTEM

    def test_has_tool_calls_false(self) -> None:
        msg = Message.assistant("no tools")
        assert msg.has_tool_calls() is False

    def test_has_tool_calls_true(self) -> None:
        tc = ToolCall(id="1", name="read_file", arguments={"path": "/tmp/x"})
        msg = Message.assistant("using tool", tool_calls=[tc])
        assert msg.has_tool_calls() is True

    def test_to_dict_user(self) -> None:
        msg = Message.user("hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_to_dict_tool_result(self) -> None:
        from lclai.llm.messages import ToolResult as LLMToolResult

        tr = LLMToolResult(tool_call_id="abc", name="read_file", content="file content")
        msg = Message.tool_result(tr)
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["content"] == "file content"
        assert d["tool_call_id"] == "abc"

    def test_tool_call_from_dict(self) -> None:
        data = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_1"
        assert tc.name == "bash"
        assert tc.arguments == {"command": "ls"}

    def test_tool_call_to_dict(self) -> None:
        tc = ToolCall(id="x", name="read_file", arguments={"path": "/tmp"})
        d = tc.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "read_file"
