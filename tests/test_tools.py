"""Tests for the tools layer — decorator, registry, executor, discovery."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from maque_core.contracts.tool import ToolSpec, CallBudget
from maque_core.tools._global_registry import clear_global, get_global_specs
from maque_core.tools.decorator import tool
from maque_core.tools.registry import ToolRegistry
from maque_core.tools.executor import ToolExecutor, InProcessBackend
from maque_core.tools.discovery import discover_tools, discover_from_path


# ============================================================
# @tool decorator tests
# ============================================================

class TestToolDecorator:
    def setup_method(self):
        clear_global()

    def test_bare_decorator_registers(self):
        @tool
        def my_func(x: str) -> str:
            """My test function."""
            return x

        specs = get_global_specs()
        assert "my_func" in specs
        assert specs["my_func"].description == "My test function."
        assert specs["my_func"].func is my_func

    def test_named_decorator(self):
        @tool(name="custom_name", description="Custom description")
        def some_func(x: str) -> str:
            return x

        specs = get_global_specs()
        assert "custom_name" in specs
        assert specs["custom_name"].description == "Custom description"

    def test_decorator_preserves_function(self):
        @tool
        def echo(text: str) -> str:
            """Echo text."""
            return text

        # The original function should still work
        assert echo("hello") == "hello"

    def test_string_param_schema(self):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"

        spec = get_global_specs()["greet"]
        props = spec.parameters["properties"]
        assert props["name"]["type"] == "string"

    def test_integer_param_schema(self):
        @tool
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        spec = get_global_specs()["double"]
        assert spec.parameters["properties"]["n"]["type"] == "integer"

    def test_float_param_schema(self):
        @tool
        def half(x: float) -> float:
            """Half a number."""
            return x / 2

        spec = get_global_specs()["half"]
        assert spec.parameters["properties"]["x"]["type"] == "number"

    def test_bool_param_schema(self):
        @tool
        def toggle(flag: bool) -> bool:
            """Toggle a flag."""
            return not flag

        spec = get_global_specs()["toggle"]
        assert spec.parameters["properties"]["flag"]["type"] == "boolean"

    def test_required_params(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        spec = get_global_specs()["add"]
        required = spec.parameters.get("required", [])
        assert "a" in required
        assert "b" in required

    def test_optional_params_not_required(self):
        from typing import Optional

        @tool
        def greet(name: str, greeting: Optional[str] = None) -> str:
            """Greet with optional custom greeting."""
            return f"{greeting or 'Hello'}, {name}!"

        spec = get_global_specs()["greet"]
        required = spec.parameters.get("required", [])
        assert "name" in required
        assert "greeting" not in required

    def test_default_params_not_required(self):
        @tool
        def repeat(text: str, times: int = 1) -> str:
            """Repeat text."""
            return text * times

        spec = get_global_specs()["repeat"]
        required = spec.parameters.get("required", [])
        assert "text" in required
        assert "times" not in required

    def test_description_from_docstring_first_line(self):
        @tool
        def multi_line(x: str) -> str:
            """First line is the description.

            This is more detail that shouldn't be in the description.
            Args:
                x: input
            """
            return x

        spec = get_global_specs()["multi_line"]
        assert spec.description == "First line is the description."

    def test_no_docstring_gives_empty_description(self):
        @tool
        def no_doc(x: str) -> str:
            return x

        spec = get_global_specs()["no_doc"]
        assert spec.description == ""

    def test_generic_list_schema(self):
        from typing import List

        @tool
        def foo(items: List[str]) -> str:
            """Test generic list."""
            return ""

        spec = get_global_specs()["foo"]
        assert spec.parameters["properties"]["items"]["type"] == "array"

    def test_generic_dict_schema(self):
        from typing import Dict, Any

        @tool
        def bar(mapping: Dict[str, Any]) -> str:
            """Test generic dict."""
            return ""

        spec = get_global_specs()["bar"]
        assert spec.parameters["properties"]["mapping"]["type"] == "object"


# ============================================================
# ToolRegistry tests
# ============================================================

class TestToolRegistry:
    def setup_method(self):
        clear_global()
        self.registry = ToolRegistry()

    def _register_echo(self):
        spec = ToolSpec(
            name="echo",
            description="Echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            func=lambda text: text,
        )
        self.registry.register(spec)
        return spec

    def test_register_tool(self):
        self._register_echo()
        assert "echo" in self.registry.tool_names
        assert len(self.registry) == 1

    def test_register_multiple_tools(self):
        self._register_echo()
        spec2 = ToolSpec(
            name="add",
            description="Add numbers",
            parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
            func=lambda a: a,
        )
        self.registry.register(spec2)
        assert len(self.registry) == 2
        assert set(self.registry.tool_names) == {"echo", "add"}

    def test_get_schemas(self):
        self._register_echo()
        schemas = self.registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"

    def test_execute_success(self):
        self._register_echo()
        obs = self.registry.execute("echo", {"text": "hello"}, "call_1")
        assert obs.tool_call_id == "call_1"
        assert obs.tool_name == "echo"
        assert obs.content == "hello"
        assert obs.is_error is False

    def test_execute_unknown_tool(self):
        obs = self.registry.execute("nonexistent", {}, "call_1")
        assert obs.is_error is True
        assert "Unknown tool" in obs.content
        assert obs.metadata["error_type"] == "unknown_tool"

    def test_execute_missing_required_param(self):
        self._register_echo()
        obs = self.registry.execute("echo", {}, "call_1")
        assert obs.is_error is True
        assert "Missing required parameter" in obs.content
        assert "text" in obs.content

    def test_execute_wrong_type_param(self):
        self._register_echo()
        obs = self.registry.execute("echo", {"text": 123}, "call_1")
        assert obs.is_error is True
        assert "expected type 'string'" in obs.content

    def test_execute_tool_exception(self):
        def bad_func(text):
            raise ValueError("something broke")

        spec = ToolSpec(
            name="bad",
            description="Bad tool",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=bad_func,
        )
        self.registry.register(spec)
        obs = self.registry.execute("bad", {"text": "hi"}, "call_1")
        assert obs.is_error is True
        assert "ValueError" in obs.content
        assert "something broke" in obs.content

    def test_execute_non_string_result(self):
        spec = ToolSpec(
            name="get_data",
            description="Get data",
            parameters={"type": "object", "properties": {}},
            func=lambda: {"key": "value", "num": 42},
        )
        self.registry.register(spec)
        obs = self.registry.execute("get_data", {}, "call_1")
        assert obs.is_error is False
        assert "key" in obs.content
        assert "42" in obs.content

    def test_register_from_specs(self):
        from maque_core.tools._global_registry import _global_tool_specs
        _global_tool_specs.clear()

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"

        count = self.registry.register_from_specs(get_global_specs())
        assert count == 1
        assert "greet" in self.registry.tool_names

    def test_get_spec(self):
        spec = self._register_echo()
        result = self.registry.get_spec("echo")
        assert result is spec
        assert self.registry.get_spec("nonexistent") is None

    def test_len(self):
        assert len(self.registry) == 0
        self._register_echo()
        assert len(self.registry) == 1

    def test_tool_names_property(self):
        assert self.registry.tool_names == []
        self._register_echo()
        assert self.registry.tool_names == ["echo"]


# ============================================================
# Budget integration tests
# ============================================================

class TestRegistryBudget:
    def test_budget_limits_execution(self):
        budget = CallBudget(max_calls=2)
        registry = ToolRegistry(budget=budget)
        spec = ToolSpec(
            name="echo",
            description="Echo",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=lambda text: text,
        )
        registry.register(spec)

        # First call — should work
        obs1 = registry.execute("echo", {"text": "1"}, "c1")
        assert obs1.is_error is False
        assert budget.used == 1

        # Second call — should work
        obs2 = registry.execute("echo", {"text": "2"}, "c2")
        assert obs2.is_error is False
        assert budget.used == 2

        # Third call — budget exhausted
        obs3 = registry.execute("echo", {"text": "3"}, "c3")
        assert obs3.is_error is True
        assert "budget" in obs3.content.lower()

    def test_no_budget_unlimited(self):
        registry = ToolRegistry()
        spec = ToolSpec(
            name="echo",
            description="Echo",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=lambda text: text,
        )
        registry.register(spec)

        for i in range(10):
            obs = registry.execute("echo", {"text": str(i)}, f"c{i}")
            assert obs.is_error is False


# ============================================================
# Hook tests
# ============================================================

class TestRegistryHooks:
    def setup_method(self):
        self.registry = ToolRegistry()
        spec = ToolSpec(
            name="echo",
            description="Echo",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=lambda text: text,
        )
        self.registry.register(spec)

    def test_before_hook_called(self):
        called = []
        def hook(name, args):
            called.append((name, args))
            return None

        self.registry.add_before_hook(hook)
        self.registry.execute("echo", {"text": "hi"}, "c1")
        assert len(called) == 1
        assert called[0][0] == "echo"
        assert called[0][1] == {"text": "hi"}

    def test_after_hook_called_on_success(self):
        called = []
        def hook(name, args, result, error):
            called.append({"name": name, "result": result, "error": error})

        self.registry.add_after_hook(hook)
        self.registry.execute("echo", {"text": "hi"}, "c1")
        assert len(called) == 1
        assert called[0]["result"] == "hi"
        assert called[0]["error"] is None

    def test_after_hook_called_on_error(self):
        # Register a tool that raises an exception during execution
        def bad_func(text):
            raise RuntimeError("execution failed")

        spec = ToolSpec(
            name="bad_tool",
            description="A tool that always fails",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            func=bad_func,
        )
        self.registry.register(spec)

        called = []
        def hook(name, args, result, error):
            called.append({"name": name, "result": result, "error": error})

        self.registry.add_after_hook(hook)
        obs = self.registry.execute("bad_tool", {"text": "hi"}, "c1")
        assert obs.is_error is True  # verify it actually failed
        assert len(called) == 1
        assert called[0]["name"] == "bad_tool"
        assert called[0]["result"] is None
        assert called[0]["error"] is not None
        assert isinstance(called[0]["error"], RuntimeError)

    def test_multiple_before_hooks(self):
        count = [0]
        def hook1(name, args):
            count[0] += 1
            return None
        def hook2(name, args):
            count[0] += 1
            return None

        self.registry.add_before_hook(hook1)
        self.registry.add_before_hook(hook2)
        self.registry.execute("echo", {"text": "hi"}, "c1")
        assert count[0] == 2


# ============================================================
# ToolExecutor tests
# ============================================================

class TestToolExecutor:
    def test_in_process_backend_success(self):
        backend = InProcessBackend(max_workers=2)
        result = backend.execute(lambda x: x * 2, {"x": 21})
        assert result["is_error"] is False
        assert result["result"] == 42
        assert "duration_ms" in result
        backend.close()

    def test_in_process_backend_exception(self):
        backend = InProcessBackend()
        def fail():
            raise ValueError("test error")
        result = backend.execute(fail, {})
        assert result["is_error"] is True
        assert "ValueError" in result["error"]
        backend.close()

    def test_tool_executor_context_manager(self):
        with ToolExecutor() as executor:
            result = executor.run(lambda: 42, {})
            assert result["result"] == 42
            assert result["is_error"] is False

    def test_tool_executor_default_timeout(self):
        executor = ToolExecutor(default_timeout=5.0)
        assert executor._default_timeout == 5.0
        executor.close()

    def test_executor_closed_returns_error(self):
        executor = ToolExecutor()
        executor.close()
        result = executor.run(lambda: 1, {})
        assert result["is_error"] is True
        assert "closed" in result["error"]

    def test_executor_with_timeout_param(self):
        with ToolExecutor() as executor:
            result = executor.run(lambda: "ok", {}, timeout=1.0)
            assert result["result"] == "ok"


# ============================================================
# Discovery tests
# ============================================================

class TestDiscovery:
    # Note: We do NOT clear_global() for discovery tests.
    # The global registry pattern relies on @tool decorators registering
    # at import time. Since modules are cached by Python, clearing and
    # re-importing won't re-run decorators. Discovery tests verify that
    # discover_tools() correctly loads from the already-populated global
    # registry into the target registry.

    def test_discover_tools_from_package(self):
        # Ensure impl modules are freshly imported so @tool decorators run.
        # Other tests may have called clear_global(), and Python's module
        # cache means re-importing doesn't re-run decorators. Use reload
        # to force re-execution.
        import importlib
        import maque_core.tools.impl.echo_tool
        import maque_core.tools.impl.add_tool
        importlib.reload(maque_core.tools.impl.echo_tool)
        importlib.reload(maque_core.tools.impl.add_tool)

        registry = ToolRegistry()
        count = discover_tools(registry, "maque_core.tools.impl")
        assert count >= 2  # echo + add
        assert "echo" in registry.tool_names
        assert "add" in registry.tool_names

    def test_discover_idempotent(self):
        registry = ToolRegistry()
        count1 = discover_tools(registry, "maque_core.tools.impl")
        count2 = discover_tools(registry, "maque_core.tools.impl")
        # Second call should find 0 new tools (already registered)
        assert count2 == 0
        assert len(registry) == count1

    def test_discover_nonexistent_package(self):
        registry = ToolRegistry()
        count = discover_tools(registry, "nonexistent.package.name")
        assert count == 0
        assert len(registry) == 0

    def test_discover_from_path(self):
        # Create a temp directory with a tool file
        with tempfile.TemporaryDirectory() as tmpdir:
            tool_file = os.path.join(tmpdir, "my_temp_tool.py")
            with open(tool_file, "w") as f:
                f.write("""
from maque_core.tools.decorator import tool

@tool
def temp_echo(text: str) -> str:
    \"\"\"A temporary echo tool.\"\"\"
    return text
""")
            registry = ToolRegistry()
            count = discover_from_path(registry, tmpdir)
            # Should find temp_echo plus any already in global registry
            assert count >= 1
            assert "temp_echo" in registry.tool_names
