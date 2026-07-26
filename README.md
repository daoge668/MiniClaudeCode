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
- 使用 Worktree 功能时需要 Git。

## 安装

```powershell
git clone https://github.com/<你的GitHub用户名>/MiniClaudeCode.git
cd MiniClaudeCode
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item .env
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
这里的 `pip` 只安装第三方依赖，不会安装 MiniClaudeCode 项目本身。

## 项目知识库（RAG）

RAG 默认是可选功能。配置项示例见 `.env.rag.example`。当前配置使用
OpenAI-compatible embedding 接口，推荐模型：

```dotenv
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
```

先启动 Docker Desktop，再启动 Milvus：

```powershell
docker compose -f docker-compose.rag.yml up -d
docker compose -f docker-compose.rag.yml ps
```

把项目私有资料放入 `resources/`。程序每次启动时只同步有变化的文件，运行期间
不会监听文件变化；要让修改后的资料生效，需要重启程序。索引和 Milvus 数据保存在
本地 `.rag/`，不会提交到 Git。程序不会自动启动或删除 Docker 容器。

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
├── code.py
├── docker-compose.rag.yml
├── resources/
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
