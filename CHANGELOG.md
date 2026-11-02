# Changelog — maque_core-framework

> v1.3.0 起项目改名为 maque_core（原名 myagent）。以下 v1.3.0 变更描述中的旧路径均已对应到新包名。

## v1.3.4 (2026-10-30)

Patch release — six accumulated fixes from v1.3.0 plus two changelog wording fixes.

### Fixes

- **Three `.bat` scripts** — `MYAGENT_*` → `MAQUE_*` environment variables. Without this, the launcher sets old variable names and `config.py` reads defaults, so the agent falls back to DashScope instead of connecting to the local llama.cpp server.
- **`kernel/loop.py`** — `_chat_with_length_retry` now wraps the retry call and `messages.pop()` in `try/finally`, so temporary messages can't leak into history if `llm.chat` raises.
- **`kernel/agent.py`** — `ask()` catches `Exception` instead of `AgentRunError` during rollback, so history is restored regardless of exception type.
- **`tools/decorator.py`** — `_type_to_schema` now handles `List[X]` and `Dict[K, V]` via `typing.get_origin`, mapping them to `array` / `object` instead of falling back to `string`.
- **`llm/__init__.py`** — removed the `Qwen3XMLFallback` docstring entry (the class never existed in code).
- **`examples/demo.py`** — Demo 4 now registers the `multiply` tool on the agent before `ask()`, so the trace no longer shows `Unknown tool: multiply`.
- **`CHANGELOG.md`** — v1.3.2 heading fixed to match body ("three cross-layer imports"); v1.3.3 test-name arithmetic clarified.

## v1.3.3 (2026-10-30)

Documentation fixes — corrected test names in TEST_REPORT.md and minor wording fixes.

### Changes

- **`TEST_REPORT.md`** — Fixed 6 fabricated test names in `test_factory.py` and `test_public_api.py` sections. 4 factory tests + 3 public_api tests = 7 names; 6 were wrong and have been corrected (1 happened to match: test_create_agent_with_tools_package).
- **`CHANGELOG.md`** — v1.3.2 entry: "two cross-layer imports" → "three cross-layer imports (AppConfig, discover_tools, ConcreteRegistry)" to match what was actually removed.

## v1.3.2 (2026-10-30)

Kernel purification — removed the cross-layer imports in `kernel/agent.py`.

### Changes

- **`kernel/agent.py`** — Removed `Agent.from_config()` classmethod and the three cross-layer imports (`AppConfig`, `discover_tools`, `ConcreteRegistry`). Kernel now has **zero** dependencies on tools/, config/, or llm/ — only contracts + kernel internals.
- **`factory.py`** — `create_agent()` is now the single entry point for config-based agent construction (no duplicate logic).
- **`app.py` / `webui.py`** — Switched from `Agent.from_config(config, llm)` to `create_agent(llm=llm, ..., tools_package=config.tools_package)`.
- **Tests** — Renamed `test_agent_from_config` → `test_create_agent` to reflect the new API.

### Kernel Purity Verification

```python
# kernel/ imports only contracts + its own files — zero cross-layer deps
from maque_core.contracts.messages import Message
from maque_core.kernel import Agent, react_loop, RunBudget, AgentRunError
# → kernel OK
```

## v1.3.1 (2026-10-30)

Patch release — documentation fixes and minor improvements after v1.3.0 rename.

### Changes

- **CHANGELOG.md** — Fixed UTF-8 encoding; all paths updated from `myagent/` to `maque_core/`
- **TEST_REPORT.md** — Title updated to `maque_core v1.3.1`; test count corrected to 318 (added `test_factory.py` and `test_public_api.py` rows)
- **tests/test_integration.py** — Replaced `MyAgent!` brand references with `maque_core!`
- **webui.py** — Error response now includes `"trace"` field (previously missing in `AgentRunError` branch)
- **README.md** — Project structure updated with `factory.py`, `errors.py`, `webui.py`, new test files, and `examples/tools/file_tool.py`

## v1.3.0 (2026-10-30)

### P0 — Error Propagation Architecture Change

**Before (v1.2):** Errors travelled through the **content channel** — a failed LLM call returned an `LLMResponse` with `error` set, and `content` carried the error text. Callers had to check `response.is_error` after every call, and the answer string and error string shared the same field.

**After (v1.3):** Errors travel through the **exception channel** — `AgentRunError` is raised on failure. The `content` field only holds real answers. This matches the existing `OpenAICompatibleProvider.chat(raise_on_error=False)` pattern at the adapter layer.

### Layered Defaults

| Layer | Function | `raise_on_error` default | Rationale |
|-------|----------|-------------------------|-----------|
| Kernel (primitive) | `react_loop()` | `False` | Backward compatible — internal primitive, caller decides |
| Facade (user-facing) | `Agent.ask()` | `True` | Clean API — users expect exceptions on failure |
| Adapter | `LLMProvider.chat()` | `False` | Low-level — gives callers the choice |

### Changes

#### 1. New: `maque_core/kernel/errors.py`
- Added `AgentRunError` exception class
- Carries `message` (human-readable) and `response` (raw `LLMResponse`, optional)
- Raised by the kernel loop when `raise_on_error=True`

#### 2. Modified: `maque_core/kernel/loop.py`
- `react_loop()` — added `raise_on_error: bool = False` parameter
- `_chat_with_length_retry()` — added `raise_on_error: bool = False`; raises `AgentRunError` when `response.error is not None` and flag is set (both initial call and length-retry combined response)
- `_force_final_answer()` — added `raise_on_error: bool = False`; same error check
- Parameter is threaded through all internal call paths

#### 3. Modified: `maque_core/kernel/agent.py`
- `ask()` — added `raise_on_error: bool = True` (default True — facade layer fails fast)
- Added history rollback: `snapshot = len(self.messages)` before the call; on `AgentRunError`, `del self.messages[snapshot:]` rolls back all messages added during the failed turn
- **Fixes P2 "history pollution"**: a failed call no longer leaves an orphan user message in the conversation history
- Error is re-raised after rollback for upper-layer handling

#### 4. Modified: `maque_core/kernel/__init__.py`
- Exports `AgentRunError`

#### 5. Modified: `app.py`
- `cmd_ask()` — catches `AgentRunError`, prints `[error] <message>` to stderr, exits with code 1
- `cmd_interactive()` — catches `AgentRunError`, prints `[error] <message>` to stderr, **does not break the loop** (user can continue the conversation)

#### 6. New: `webui.py`
- Web UI / JSON API layer with `RecordingProvider` for trace display
- Explicitly catches `AgentRunError` and returns `{"ok": false, "error": "...", "hint": "..."}`
- `RecordingProvider` records all LLM calls for trace display; error detection is handled by the exception channel (no `last_error` flag on the provider)

#### 7. New: `tests/test_error_propagation.py`
- 5 tests covering the new error propagation architecture
- All 311 tests pass (306 existing + 5 new)

#### 8. New: `maque_core/factory.py` + `tests/test_factory.py`
- `create_agent()` factory function as unified entry point
- Auto-configures LLM from env vars (`MAQUE_LLM_*`)
- Supports `tools_package` for auto tool discovery
- 4 factory tests

#### 9. New: `tests/test_public_api.py`
- 3 tests validating public API surface (`__all__` = 14 names)

### Backward Compatibility

- `react_loop()` default behaviour unchanged — existing callers that don't pass `raise_on_error` get the same error-in-response behaviour
- `Agent.ask("question")` positional calls still work — `raise_on_error` is a keyword parameter with a default
- `LLMResponse.error` field is preserved (still set on failure at the adapter layer)
- All 306 existing tests pass without modification
