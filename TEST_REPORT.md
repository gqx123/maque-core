# Test Report — maque_core v1.3.4

**Total tests: 320** — **100% passed** (0 failed, 0 skipped, 0 errors)

## Test Distribution by Module

| Module | Tests | Description |
|--------|-------|-------------|
| `tests/test_llm.py` | 70 | LLM provider layer (mock + OpenAI-compatible adapter) |
| `tests/test_contracts.py` | 67 | Core data contracts (messages, LLMResponse, ToolCall, etc.) |
| `tests/test_kernel.py` | 61 | Kernel layer (ReAct loop, Agent, budget) |
| `tests/test_llm_resilience.py` | 54 | LLM resilience patterns (retry, fallback, error classification) |
| `tests/test_tools.py` | 43 | Tool registry, decorator, executor, discovery |
| `tests/test_integration.py` | 13 | End-to-end integration tests |
| `tests/test_error_propagation.py` | 5 | P0 error propagation via exception channel |
| `tests/test_factory.py` | 4 | `create_agent` factory function |
| `tests/test_public_api.py` | 3 | Public API surface snapshot |

## New Tests (v1.3.0)

`tests/test_error_propagation.py` — 5 tests covering the new error architecture:

1. **`test_react_loop_raises_on_error_when_enabled`** — `react_loop(raise_on_error=True)` raises `AgentRunError` with the raw `LLMResponse` attached
2. **`test_react_loop_default_keeps_legacy_behavior`** — `react_loop` default (`raise_on_error=False`) preserves backward compatibility
3. **`test_agent_ask_raises_on_llm_error`** — `Agent.ask()` raises `AgentRunError` by default (facade layer)
4. **`test_agent_ask_rolls_back_history_on_error`** — Failed `ask()` rolls back conversation history; no orphan user message (P2 fix)
5. **`test_agent_ask_normal_still_returns_str`** — Normal success path is unaffected

`tests/test_factory.py` — 4 tests for the `create_agent()` factory:

1. **`test_create_agent_no_args`** — Factory creates an agent with all defaults (no explicit args)
2. **`test_create_agent_with_tools_package`** — Tools auto-discovery via `tools_package` param (string form)
3. **`test_create_agent_with_injected_llm`** — Factory accepts a pre-built LLM provider via `llm=`
4. **`test_create_agent_model_override`** — Explicit `model=` param overrides the env default

`tests/test_public_api.py` — 3 tests validating the public API surface:

1. **`test_public_api_is_exactly_expected`** — `__all__` matches the expected set exactly (no more, no less)
2. **`test_every_exported_name_exists`** — Every name in `__all__` is importable from the package
3. **`test_internal_modules_not_in_all`** — Internal modules (kernel, llm, tools, etc.) are not in `__all__`

## Compatibility Verification

- ✅ All 306 existing tests from v1.2.0 continue to pass
- ✅ `react_loop` default behaviour unchanged (no raise)
- ✅ `Agent.ask("q")` positional call still works (new param is keyword-only with default)
- ✅ `AgentRunError` exported from `maque_core.kernel`
