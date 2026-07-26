# MiniClaudeCode

MiniClaudeCode 是一个基于 Anthropic API 的模块化命令行编码 Agent。它提供
文件和命令工具、任务依赖、Git Worktree、技能加载、权限钩子、子代理与队友、
上下文压缩、后台任务、Cron 调度、记忆以及可动态发现的 MCP 工具。

> MiniClaudeCode 是独立开发的开源项目，不隶属于 Anthropic，也未获得
> Anthropic 官方认可。“Claude”是 Anthropic 的商标。

## 功能

- 通过统一权限管线执行命令和文件操作。
- 使用任务图、队友线程与隔离 Worktree 协作。
- 自动控制上下文大小，并在需要时保存转录记录。
- 支持后台任务、持久化 Cron 和长期记忆文件。
- 通过 MCP 注册表在运行期间发现和接入工具。
- 导入 Python 包时不访问网络、不创建目录、不启动线程。

## 环境要求

- Python 3.10 或更高版本。
- Anthropic API Key 和可用模型 ID。
- 使用 Worktree 功能时需要 Git。

## 安装

```powershell
git clone https://github.com/<你的GitHub用户名>/MiniClaudeCode.git
cd MiniClaudeCode
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
MODEL_ID=你的模型ID
ANTHROPIC_API_KEY=你的API密钥

# 可选
# FALLBACK_MODEL_ID=备用模型ID
# ANTHROPIC_BASE_URL=https://api.anthropic.com
```

`.env` 包含敏感信息，已被 `.gitignore` 排除，禁止提交到 GitHub。

## 启动

```powershell
.\.venv\Scripts\python.exe code.py
```

也可以使用包入口或安装后的命令：

```powershell
.\.venv\Scripts\python.exe -m mini_claude_code
mini-claude-code
```

默认把项目目录作为工作区。使用其他目录：

```powershell
mini-claude-code --workdir C:\path\to\workspace
```

输入问题后按回车发送；输入 `q`、`exit` 或空行退出。

## 目录结构

```text
MiniClaudeCode/
├── code.py
├── pyproject.toml
├── .env.example
├── mini_claude_code/
│   ├── cli.py
│   ├── runtime.py
│   ├── tools.py
│   ├── agents.py
│   └── ...
└── tests/
```

运行过程中可能产生以下本地数据，它们默认不会提交到 Git：

- `.tasks/`：任务状态。
- `.mailboxes/`：队友消息。
- `.worktrees/`：隔离工作树和事件。
- `.transcripts/`：上下文压缩前的会话记录。
- `.memory/MEMORY.md`：用户维护的长期记忆。
- `.task_outputs/`：较大的工具输出。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m compileall -q mini_claude_code code.py
.\.venv\Scripts\python.exe code.py --help
```

自动测试使用假模型客户端，不发送真实 API 请求。

## 发布到 GitHub

确认 `.env` 和运行数据未被跟踪后：

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial release"
git remote add origin https://github.com/<你的GitHub用户名>/MiniClaudeCode.git
git push -u origin main
```

公开仓库之前还应选择并添加许可证。没有 `LICENSE` 时，其他人可以查看代码，
但默认没有复制、修改或再发布代码的许可。
