"""Tool Executor — pluggable execution backend for tool calls.

Supports:
- InProcessBackend: runs tools in-process via ThreadPoolExecutor,
  with timeout control.
- ExecutionBackend protocol: implement your own (e.g. subprocess,
  remote, sandboxed).
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for tool execution backends."""

    def execute(
        self,
        func: Callable,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute a function with the given arguments.

        Returns a dict with keys:
            - result: the return value (on success)
            - error: the error message (on failure)
            - is_error: bool
            - duration_ms: execution time in milliseconds
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...


class InProcessBackend:
    """Executes tools in-process using a thread pool.

    Provides timeout control via ThreadPoolExecutor.
    """

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute(
        self,
        func: Callable,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        future = self._executor.submit(func, **args)
        try:
            result = future.result(timeout=timeout)
            duration_ms = int((time.time() - start) * 1000)
            return {
                "result": result,
                "error": None,
                "is_error": False,
                "duration_ms": duration_ms,
            }
        except FutureTimeout:
            duration_ms = int((time.time() - start) * 1000)
            return {
                "result": None,
                "error": f"Execution timed out after {timeout}s",
                "is_error": True,
                "duration_ms": duration_ms,
                "exception": TimeoutError(f"Execution timed out after {timeout}s"),
            }
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            return {
                "result": None,
                "error": f"{type(e).__name__}: {e}",
                "is_error": True,
                "duration_ms": duration_ms,
                "traceback": traceback.format_exc(),
                "exception": e,  # original exception object for hooks/inspection
            }

    def close(self) -> None:
        self._executor.shutdown(wait=False)


class ToolExecutor:
    """High-level tool executor with a pluggable backend.

    Usage::

        with ToolExecutor() as executor:
            result = executor.run(my_func, {"x": 1}, timeout=5.0)
    """

    def __init__(
        self,
        backend: Optional[ExecutionBackend] = None,
        default_timeout: Optional[float] = 30.0,
    ):
        self._backend = backend or InProcessBackend()
        self._default_timeout = default_timeout
        self._closed = False

    def run(
        self,
        func: Callable,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a function with the configured backend.

        Args:
            func: The callable to execute.
            args: Keyword arguments dict.
            timeout: Timeout in seconds. Uses default_timeout if None.

        Returns:
            Execution result dict (see ExecutionBackend.execute).
        """
        if self._closed:
            return {
                "result": None,
                "error": "Executor has been closed",
                "is_error": True,
                "duration_ms": 0,
            }
        actual_timeout = timeout if timeout is not None else self._default_timeout
        return self._backend.execute(func, args, timeout=actual_timeout)

    def close(self) -> None:
        """Close the executor and release resources."""
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> "ToolExecutor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
