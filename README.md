# MiniClaudeCode

MiniClaudeCode 是一个基于 Anthropic API 的模块化命令行编码 Agent。它提供
文件和命令工具、任务依赖、Git Worktree、技能加载、权限钩子、子代理与队友、
上下文压缩、后台任务、Cron 调度、记忆以及可动态发现的 MCP 工具。

## 功能

- 通过统一权限管线执行命令和文件操作。
- 使用任务图、队友线程与隔离 Worktree 协作。
- 自动控制上下文大小，并在需要时保存转录记录。
- 支持后台任务、持久化 Cron 和长期记忆文件。
- 通过 MCP 注册表在运行期间发现和接入工具。
- 启动时增量索引 `resources/`，由 Agent 自主调用项目知识检索工具。
- 导入 Python 包时不访问网络、不创建目录、不启动线程。

## 环境要求

- Python 3.10 或更高版本。
- Anthropic API Key 和可用模型 ID。
- 启用 RAG 时需要 Docker Desktop（或兼容的 Docker Engine 与 Compose）。
- 使用 Worktree 功能时需要 Git。

## 安装

```powershell
git clone https://github.com/daoge668/MiniClaudeCode.git
cd MiniClaudeCode
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`。主模型配置必填，RAG 配置可按需启用：

```dotenv
MODEL_ID=你的模型ID
ANTHROPIC_API_KEY=你的API密钥

# 主模型可选项
# FALLBACK_MODEL_ID=备用模型ID
# ANTHROPIC_BASE_URL=https://api.anthropic.com

# 项目知识库
RAG_ENABLED=true
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=root:Milvus
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=你的千问API密钥
EMBEDDING_MODEL=qwen3.7-text-embedding
RERANK_ENABLED=true
RERANK_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_MODEL=qwen3-vl-rerank
RERANK_CANDIDATES=20
RERANK_TIMEOUT_SECONDS=30
```

`.env.example` 是唯一的配置模板，不包含真实密钥；`.env` 包含本机密钥并已被
`.gitignore` 排除，禁止提交到 GitHub。这里的 `pip` 只安装第三方依赖，不会
安装 MiniClaudeCode 项目本身。

## 项目知识库（RAG）

RAG 默认是可选功能，使用 OpenAI-compatible embedding 接口。当前推荐
`qwen3.7-text-embedding` 生成向量，`qwen3-vl-rerank` 对召回结果重排。

先启动 Docker Desktop，再启动 Milvus：

```powershell
docker compose -f docker-compose.rag.yml up -d
docker compose -f docker-compose.rag.yml ps
```

把项目私有资料放入 `resources/`。程序每次启动时只同步有变化的文件，运行期间
不会监听文件变化；要让修改后的资料生效，需要重启程序。索引和 Milvus 数据保存在
本地 `.rag/`，不会提交到 Git。程序不会自动启动或删除 Docker 容器。

首次使用时，等待 `docker compose ... ps` 中三个服务都显示为健康，再启动
MiniClaudeCode。启动日志出现 RAG 同步统计后，主 Agent、focused subagent 和
teammate 会共享同一份索引快照，并可自主调用 `search_project_knowledge`。

Milvus 不可用时，RAG 工具不会注册，其他 Agent 功能仍可运行。embedding 服务暂时
不可用但已有集合时，检索会退化为本地 BM25。启用重排后，系统会把 RRF 或 BM25
召回的前 20 段交给 `qwen3-vl-rerank`，再返回前 5 段；重排服务不可用时会自动
使用原始检索顺序。`RERANK_API_KEY` 未配置时会复用 `EMBEDDING_API_KEY`。

## 启动

```powershell
.\.venv\Scripts\python.exe code.py
```

也可以使用模块入口：

```powershell
.\.venv\Scripts\python.exe -m mini_claude_code
```

默认把项目目录作为工作区。使用其他目录：

```powershell
.\.venv\Scripts\python.exe code.py --workdir C:\path\to\workspace
```

输入问题后按回车发送；输入 `q`、`exit` 或空行退出。

## 目录结构

```text
MiniClaudeCode/
├── .env.example
├── code.py
├── docker-compose.rag.yml
├── resources/
│   └── rag-usage.md
├── requirements.txt
├── mini_claude_code/
│   ├── rag/
│   ├── cli.py
│   ├── runtime.py
│   ├── tools.py
│   ├── agents.py
│   └── ...
```

运行过程中可能产生以下本地数据，它们默认不会提交到 Git：

- `.tasks/`：任务状态。
- `.mailboxes/`：队友消息。
- `.worktrees/`：隔离工作树和事件。
- `.transcripts/`：上下文压缩前的会话记录。
- `.memory/MEMORY.md`：用户维护的长期记忆。
- `.task_outputs/`：较大的工具输出。
