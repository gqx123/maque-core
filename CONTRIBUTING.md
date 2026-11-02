# 贡献指南

感谢你对 maque_core 的关注！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告 Bug
- 💡 提出新功能建议
- 📖 改进文档
- 🔧 提交代码修复或新功能
- 💬 参与社区讨论

## 快速开始

### 1. Fork 并克隆仓库

```bash
# Fork 本仓库后，克隆你的 fork
git clone https://github.com/your-username/maque-core.git
cd maque-core
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install pytest  # 运行测试需要
```

### 3. 运行测试

确保所有测试通过：

```bash
pytest tests/ -v
```

### 4. 创建分支

```bash
git checkout -b feature/your-feature-name
```

## 提交规范

### 代码风格

- 遵循 [PEP 8](https://peps.python.org/pep-0008/) 代码风格
- 使用类型注解（Type Hints）
- 文档字符串使用 Google 风格

### Commit Message

使用清晰的描述性提交信息，建议格式：

```
<type>: <subject>

<body>

<footer>
```

类型（type）：
- `feat` — 新功能
- `fix` — Bug 修复
- `docs` — 文档变更
- `style` — 代码格式调整（不影响功能）
- `refactor` — 重构
- `test` — 测试相关
- `chore` — 构建/工具/依赖等维护类变更

### Pull Request 流程

1. 确保你的代码通过所有测试
2. 为新功能添加相应的测试用例
3. 更新相关文档（README、CHANGELOG 等）
4. 提交 PR，详细描述你的改动
5. 等待 Code Review

## 报告 Bug

在提交 Bug 报告时，请包含以下信息：

- **操作系统** 和 Python 版本
- **maque_core 版本**
- **复现步骤**（越详细越好）
- **预期行为** 和 **实际行为**
- **错误日志** 或堆栈跟踪

[提交 Bug →](https://github.com/gqx123/maque-core/issues/new)

## 功能建议

有新想法？欢迎在 [Discussions](https://github.com/gqx123/maque-core/discussions) 里分享，或者直接开一个 Feature Request Issue。

## 行为准则

参与本项目请遵守开源社区的基本礼仪：

- 尊重不同的观点和经验
- 友善地接受建设性批评
- 关注对社区最有利的事情
- 对其他成员保持同理心

## 问题？

如果有任何疑问，欢迎在 [Discussions](https://github.com/gqx123/maque-core/discussions) 中提问，我们会尽快回复。

再次感谢你的贡献！🎉
