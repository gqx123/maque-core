# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.4] - 2026-11-02

### Added
- Initial public release on GitHub
- MIT License
- Full test suite (300+ tests)
- Web UI with JSON API
- ReAct loop with budget protection
- Auto tool discovery via @tool decorator
- OpenAI-compatible LLM provider with error classification
- Protocol-based layered architecture

### Changed
- Project renamed from "myagent" to "maque_core" (since v1.3.0)

---

## [1.3.0] - 2026-10-30

### Added
- Reasoning / Chain-of-Thought extraction support
- LLM error classification (auth, rate_limit, network, timeout, etc.)
- `finish_reason` field with truncation detection
- `max_tokens` configuration support
- `ToolExecutor` with pluggable execution backends

### Changed
- Renamed from "myagent" to "maque_core"
- Improved error propagation in ReAct loop
