# maque_core — 轻量级可扩展 Agent 框架

> v1.3.0 起项目由 myagent 改名为 maque_core。

一个基于 Python 的轻量级 Agent 框架，采用分层架构设计，支持 ReAct（推理 + 行动）循环、工具自动发现、预算保护等核心能力。

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                   app.py / CLI                  │  应用入口
├─────────────────────────────────────────────────┤
│              kernel (Agent / ReAct)             │  内核层：推理+行动循环
├──────────────────┬──────────────────────────────┤
│   llm (Adapter)  │      tools (Registry)        │  能力层：LLM + 工具
├──────────────────┴──────────────────────────────┤
│              contracts (Protocol)               │  契约层：接口定义 + 数据结构
└─────────────────────────────────────────────────┘
```

**设计原则：**
- **面向协议编程**：各层依赖抽象（Protocol），而非具体实现，方便替换
- **全局注册器模式**：`@tool` 装饰器在导入时注册到全局字典，解决模块缓存导致的工具发现问题
- **双重预算保护**：最大迭代次数 + 最大工具调用数，防止 Agent 失控
- **可插拔执行后端**：工具执行支持线程池、子进程、远程等多种后端

## 快速开始

### 安装

```bash
cd maque_core-project
pip install -r requirements.txt
```

### 运行 Demo

```bash
python examples/demo.py
```

### 交互式使用

```bash
# 需要先配置 API Key
export MAQUE_LLM_API_KEY=your-api-key
python app.py -i
```

### 单次提问

```bash
python app.py "What is 2 + 2?"
```

### 列出所有工具

```bash
python app.py --list-tools
```

## 项目结构

```
maque_core-project/
├── maque_core/
│   ├── __init__.py
│   ├── config.py              # 配置管理（支持环境变量）
│   ├── factory.py             # create_agent() 工厂函数（统一入口）
│   ├── contracts/             # 契约层：接口 + 数据结构
│   │   ├── messages.py        # Role / Message / ToolCall / LLMResponse / Observation
│   │   ├── llm.py             # LLMProvider Protocol
│   │   └── tool.py            # ToolSpec / ToolRegistry Protocol / CallBudget
│   ├── llm/                   # LLM 适配器层
│   │   ├── openai_compatible.py  # OpenAI 兼容 API 适配器
│   │   ├── errors.py          # LLMError 异常分类
│   │   └── mock.py            # Mock LLM（用于测试）
│   ├── tools/                 # 工具层
│   │   ├── _global_registry.py # 全局注册器（解决模块缓存问题）
│   │   ├── decorator.py       # @tool 装饰器
│   │   ├── registry.py        # ToolRegistry（注册 + 执行 + 钩子）
│   │   ├── executor.py        # ToolExecutor（可插拔执行后端）
│   │   ├── discovery.py       # 工具自动发现
│   │   └── impl/              # 内置工具实现
│   │       ├── echo_tool.py
│   │       └── add_tool.py
│   └── kernel/                # 内核层
│       ├── budget.py          # RunBudget（双重预算保护）
│       ├── loop.py            # ReAct 循环
│       ├── errors.py          # AgentRunError 异常
│       └── agent.py           # Agent 高层接口
├── tests/                     # 测试
│   ├── test_contracts.py
│   ├── test_llm.py
│   ├── test_llm_resilience.py
│   ├── test_tools.py
│   ├── test_kernel.py
│   ├── test_integration.py
│   ├── test_error_propagation.py
│   ├── test_factory.py
│   └── test_public_api.py
├── examples/
│   ├── demo.py                # 端到端 Demo
│   └── tools/                 # 示例工具（不在内核中）
│       └── file_tool.py
├── app.py                     # CLI 入口
├── webui.py                   # Web UI / JSON API
├── requirements.txt
└── README.md
```

## 如何添加新工具

只需 3 步：

1. 在 `maque_core/tools/impl/` 下创建新的 `.py` 文件
2. 用 `@tool` 装饰器标记你的函数
3. 自动发现会在启动时加载它

```python
from maque_core.tools import tool

@tool
def weather(city: str) -> str:
    """查询指定城市的天气。"""
    # 你的实现...
    return f"{city}：晴，25°C"
```

框架会自动：
- 从类型注解生成 JSON Schema
- 识别必填参数（无默认值的参数）
- 注册到全局工具注册表
- 在 Agent 启动时自动发现

## 如何替换 LLM 后端

框架使用 `LLMProvider` Protocol，任何实现了 `chat()` 方法的类都可以作为 LLM 后端。

### 使用 OpenAI 兼容 API（默认）

```python
from maque_core.llm import OpenAICompatibleProvider

llm = OpenAICompatibleProvider(
    model="qwen-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your-key",
)
```

### 自定义 LLM 后端

```python
from maque_core.contracts.llm import LLMProvider
from maque_core.contracts.messages import LLMResponse, Message
from typing import List, Optional, Any

class MyLLM:
    def chat(self, messages: List[Message], tools=None, **kwargs) -> LLMResponse:
        # 你的实现...
        return LLMResponse(content="Hello!")
```

## 配置说明

所有配置都可以通过环境变量覆盖（前缀 `MAQUE_`）：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MAQUE_LLM_MODEL` | `qwen-turbo` | 模型名称 |
| `MAQUE_LLM_BASE_URL` | DashScope 兼容模式 | API 地址 |
| `MAQUE_LLM_API_KEY` | 空 | API 密钥 |
| `MAQUE_LLM_TEMPERATURE` | `0.7` | 采样温度 |
| `MAQUE_LLM_TIMEOUT` | `60` | 请求超时（秒） |
| `MAQUE_LLM_MAX_TOKENS` | `2048` | 单次生成的最大 token 数 |
| `MAQUE_MAX_ITERATIONS` | `10` | 最大推理轮次 |
| `MAQUE_MAX_TOOL_CALLS` | 无限制 | 最大工具调用次数 |
| `MAQUE_VERBOSE` | `false` | 详细输出 |
| `MAQUE_TOOLS_PACKAGE` | `maque_core.tools.impl` | 工具包路径 |

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_contracts.py -v
pytest tests/test_tools.py -v
pytest tests/test_kernel.py -v
pytest tests/test_integration.py -v

# 统计测试数量
pytest tests/ --collect-only -q
```

## LLM 错误处理与截断识别

LLM 调用失败时**不会再伪装成正常回答**。`LLMResponse` 新增两个可选字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `error` | `Optional[str]` | 调用失败时为错误描述，`None` 表示真实模型输出。也可用 `response.is_error` 判断 |
| `finish_reason` | `Optional[str]` | 模型停止原因，原样透传：`stop` / `length` / `tool_calls` / `content_filter`。`length` 表示被 token 上限截断，也可用 `response.is_truncated` 判断 |

失败时 `content` 仍保留错误文本（便于展示），但**上层必须判断 `error`，不能把 `content` 当答案**。

### 两种失败处理模式

```python
from maque_core.llm import OpenAICompatibleProvider, LLMError, LLMAuthError

llm = OpenAICompatibleProvider(model="qwen-turbo", api_key="your-key")

# 默认（raise_on_error=False）——兼容既有调用方：返回带 error 字段的响应
resp = llm.chat(messages)
if resp.is_error:
    ...  # 重试或告警

# raise_on_error=True——直接抛出分类异常
try:
    resp = llm.chat(messages, raise_on_error=True)
except LLMAuthError:      # 401 / 403，重试无意义
    ...
except LLMError as err:   # 其余情况
    if err.retryable:
        ...  # 退避重试
```

### 异常分类（`maque_core/llm/errors.py`）

| 异常 | `category` | `retryable` | 触发场景 |
|------|-----------|-------------|---------|
| `LLMNetworkError` | `network` | ✅ | 连接失败、DNS、代理 |
| `LLMTimeoutError` | `timeout` | ✅ | 连接 / 读 / 写超时（`NetworkError` 的子类，但可区分） |
| `LLMHTTPError` | `http` | ❌ | 其余非 2xx |
| `LLMAuthError` | `auth` | ❌ | 401 / 403 鉴权失败 |
| `LLMRateLimitError` | `rate_limit` | ✅ | 429 限流 |
| `LLMServerError` | `server` | ✅ | 5xx 上游故障 |
| `LLMResponseError` | `response` | ❌ | 响应无法解析 / 结构异常 |

所有异常携带 `status_code`、`original`（原始异常）与 `to_dict()`，便于结构化日志。

### max_tokens

请求体现在会带上 `max_tokens`，不再依赖服务端默认值：

```python
llm = OpenAICompatibleProvider(max_tokens=2048)  # 构造时设定默认值
llm.chat(messages, max_tokens=512)               # 单次覆盖，不改动默认值
llm = OpenAICompatibleProvider(max_tokens=None)  # 显式关闭，回到服务端默认
```

优先级：`chat(max_tokens=...)` > `chat(**kwargs)` > 构造参数；非正整数与 `bool` 会被忽略（`True` 是 `int` 子类，不会被当成 `1`）。

## 模块职责

| 模块 | 职责 | 关键类/函数 |
|------|------|------------|
| **contracts** | 定义所有层之间的接口契约和数据结构 | `Message`, `LLMProvider`, `ToolRegistry`, `ToolSpec` |
| **llm** | LLM 后端适配器，将不同的 LLM API 统一成 LLMProvider 接口，并对失败做分类 | `OpenAICompatibleProvider`, `MockLLM`, `LLMError` |
| **tools** | 工具注册、发现、执行和验证 | `@tool`, `ToolRegistry`, `discover_tools()`, `ToolExecutor` |
| **kernel** | Agent 核心循环，管理 ReAct 流程和预算 | `Agent`, `react_loop()`, `RunBudget` |
| **app** | 应用入口，CLI 命令行界面 | `app.py`, `examples/demo.py` |

## 安全机制

- **预算保护**：双重限制（最大迭代 + 最大工具调用），防止 Agent 无限循环
- **参数验证**：工具执行前验证参数类型和必填项
- **错误隔离**：工具执行错误被捕获并返回给 LLM，不会导致 Agent 崩溃
- **超时控制**：工具执行支持超时限制（通过 ToolExecutor）
- **钩子系统**：支持 before/after 钩子，可用于日志、审计、限流等

## 思考内容（Reasoning）支持

框架支持从 LLM 响应中提取思考内容（Chain-of-Thought / Reasoning），并以 `latest_reasoning` 属性暴露给调用方。

### 核心特性

- **三级提取策略**：按优先级依次尝试 `reasoning_content`（Qwen3/DashScope 原生）→ `reasoning`（部分 API）→ 内联 `<think>...</think>` 标签剥离
- **只保留最新**：`agent.latest_reasoning` 每次 LLM 调用后覆盖更新，只保留最近一轮的思考内容，不会累积
- **不进入对话历史**：思考内容不会被添加到 messages 中，避免污染对话上下文和增加 token 消耗
- **独立于回答**：思考内容与最终回答分离，`content` 是纯文本回答，`reasoning` 是思考过程

### 使用方式

```python
from maque_core.kernel import Agent

agent = Agent(llm=my_llm, registry=my_registry)
answer = agent.ask("Explain quantum computing.")

# 访问最新一轮的思考内容
print("思考过程:", agent.latest_reasoning)
print("最终回答:", answer)
```

### 多轮工具调用场景

在一次包含多轮工具调用的 `ask()` 中，`latest_reasoning` 只保留**最后一轮** LLM 调用的思考内容：

```python
# 假设场景：用户问 (10 + 5) * 3
# 第 1 轮：LLM 思考 → 调用 add(10, 5) → reasoning = "先算加法"
# 第 2 轮：LLM 思考 → 调用 multiply(15, 3) → reasoning = "再算乘法"
# 第 3 轮：LLM 给出最终答案 → reasoning = "结果是 45"

answer = agent.ask("What is (10 + 5) * 3?")
print(agent.latest_reasoning)  # 只输出 "结果是 45"，前两轮的思考不保留
```

### 数据结构

`LLMResponse` 新增 `reasoning: Optional[str]` 字段：

```python
@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None  # 新增：思考内容
    model: Optional[str] = None
```

## 多轮对话（Multi-turn）

框架原生支持多轮对话记忆——每次 `agent.ask()` 会自动带上之前的对话历史，无需手动管理。

### 基本用法

```python
from maque_core.kernel import Agent

agent = Agent(llm=my_llm, registry=my_registry)

# 第一轮
answer1 = agent.ask("What's the capital of France?")
# → "Paris"

# 第二轮：自动带上第一轮的上下文
answer2 = agent.ask("What's its population?")
# → 模型知道 "its" 指代 Paris
```

### 重置对话

用 `reset()` 清空历史，开始全新对话：

```python
agent.reset()
answer = agent.ask("Let's talk about something else.")
# 此时历史中只有 system prompt + 这一轮
```

### REPL 模式下的多轮对话

交互式模式（`-i` / `--interactive`）默认启用多轮记忆。可用命令：

| 命令 | 说明 |
|------|------|
| `/reset` | 重置对话，清空历史 |
| `/tools` | 列出所有可用工具 |
| `exit` / `quit` / `q` | 退出交互模式 |

## 历史裁剪策略

多轮对话会不断累积上下文，最终可能超过模型的上下文窗口。框架内置自动裁剪机制。

### 工作原理

- 配置项 `max_history_turns`（默认 20 轮）控制保留多少轮对话
- 每次 `ask()` 开头自动检查并裁剪，保留 system prompt + 最近 N 轮完整对话
- 裁剪单位是"完整轮次"（user → assistant → tool calls → tool results 整体），不会截断半轮
- system prompt 永远保留在第 0 位

### 配置方式

```python
# 构造时设置
agent = Agent(llm=my_llm, registry=my_registry, max_history_turns=10)

# 或通过环境变量
export MAQUE_MAX_HISTORY_TURNS=10
```

设为 0 或负数时禁用裁剪（不推荐，上下文可能爆掉）。

### 裁剪示意

```
max_history_turns = 3，当前有 5 轮对话：

[system] [U1] [A1] [U2] [A2+tools] [U3] [A3] [U4] [A4] [U5] [A5]
                          ↑                        ↑
                     裁剪掉前 2 轮            保留最近 3 轮 + 新轮次

裁剪后：
[system] [U3] [A3] [U4] [A4] [U5] [A5]  +  新的 U6
```

## Reasoning 入历史策略

默认情况下，每轮的 reasoning 只保存在 `latest_reasoning` 中，不进入对话历史。但多轮对话时，模型后续轮次可能需要看到之前的推理依据才能保持连贯。

### "裁剪后带上"策略

框架采用 **include_reasoning_in_history** 开关（默认 `True`）：

- **开启时**：每轮的 reasoning 作为 assistant 消息的一部分存入历史（`Message.reasoning` 字段）
- **关闭时**：reasoning 只保留在 `latest_reasoning`，不进历史（省 token）
- 历史裁剪时，reasoning 随消息一起被裁剪
- 发送给 LLM 时，reasoning 通过 `reasoning_content` 字段传递（兼容 Qwen3 / DashScope 等模型）

### 配置方式

```python
# 开启（默认）
agent = Agent(llm=my_llm, registry=my_registry, include_reasoning_in_history=True)

# 关闭（省 token）
agent = Agent(llm=my_llm, registry=my_registry, include_reasoning_in_history=False)

# 或通过环境变量
export MAQUE_INCLUDE_REASONING_IN_HISTORY=false
```

### latest_reasoning 始终可用

无论是否开启 reasoning 入历史，`agent.latest_reasoning` 始终指向最后一轮的思考内容：

```python
agent = Agent(llm=my_llm, registry=my_registry, include_reasoning_in_history=False)
answer = agent.ask("Explain photosynthesis.")
print(agent.latest_reasoning)  # 仍然可以访问最后一轮的思考
```

### Message 数据结构

`Message` 新增 `reasoning` 字段：

```python
@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    reasoning: Optional[str] = None  # 新增：该消息关联的思考内容
```

## 输出截断自动续答（finish_reason）

当 LLM 因为 token 限制截断输出（`finish_reason == "length"`）时，ReAct 循环会自动追加一条"请继续"的系统提示并重试一次，让模型把被截断的回答补完。

### 行为说明

- 仅在 `finish_reason == "length"` 且还有迭代预算时触发
- 最多重试 1 次，防止无限循环
- 两轮内容自动拼接，调用方拿到的是完整答案
- 工具调用阶段的截断不触发重试（只在纯文本输出时生效）

### 预算耗尽补 tool 消息

当工具调用预算耗尽时，框架会为未执行的 tool_call 补充一条对应的 tool 消息（说明预算耗尽），保证消息历史中 `tool_calls` 和 `tool_result` 一一对应，避免某些模型 API 因结构不匹配报错。

## 配置说明（完整）

所有配置都可以通过环境变量覆盖（前缀 `MAQUE_`）：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MAQUE_LLM_MODEL` | `qwen-turbo` | 模型名称 |
| `MAQUE_LLM_BASE_URL` | DashScope 兼容模式 | API 地址 |
| `MAQUE_LLM_API_KEY` | 空 | API 密钥 |
| `MAQUE_LLM_TEMPERATURE` | `0.7` | 采样温度 |
| `MAQUE_LLM_TIMEOUT` | `60` | 请求超时（秒） |
| `MAQUE_LLM_MAX_TOKENS` | `2048` | 单次生成的最大 token 数 |
| `MAQUE_MAX_ITERATIONS` | `10` | 单次 ask 内最大推理轮次 |
| `MAQUE_MAX_TOOL_CALLS` | 无限制 | 单次 ask 内最大工具调用次数 |
| `MAQUE_MAX_HISTORY_TURNS` | `20` | 多轮对话最大保留轮数 |
| `MAQUE_INCLUDE_REASONING_IN_HISTORY` | `true` | 是否将 reasoning 存入对话历史 |
| `MAQUE_VERBOSE` | `false` | 详细输出 |
| `MAQUE_TOOLS_PACKAGE` | `maque_core.tools.impl` | 工具包路径 |

## License

MIT
