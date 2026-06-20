# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

粮食仓储知识库 RAG 助手 —— 基于文档检索增强生成的智能问答系统。上传粮食仓储相关 PDF/DOCX 文档后，通过混合检索（BM25 + 向量 + RRF + Cross-Encoder 精排）召回相关片段，由 DeepSeek LLM 生成回答。

## 常用命令

```bash
# 启动 FastAPI 服务（开发模式）
uv run uvicorn src.api:app --reload --port 8000

# 直接运行
uv run python src/api.py

# 文档入库（增量，跳过未变更文件）
uv run python src/ingest.py

# 文档入库（全量重建，丢弃旧 ChromaDB 数据）
# ⚠️ 全量重建前必须停止 API 服务，否则多进程同时访问 ChromaDB 会导致索引损坏
uv run python src/ingest.py --full

# 测试模式（仅处理前 10 个文档，验证入库流程）
uv run python src/ingest.py --test

# 入库并测试检索
uv run python src/ingest.py --query "粮食安全水分标准"

# 检索评估：向量检索 vs 混合检索对比（MRR / NDCG / Hit@k）
uv run python eval/eval_retrieval.py
uv run python eval/eval_retrieval.py --top_k 3

# TTFT 评估：流式 vs 非流式首字延迟对比
uv run python eval/eval_ttft.py

# Redis 缓存性能测试：Redis vs MySQL 读取延迟对比
uv run python eval/eval_redis.py

# Ragas 语义评估（LLM-as-a-Judge，100 题全量）
uv run python eval/eval_ragas.py

# Ragas 快速验证（前 10 题）
uv run python eval/eval_ragas.py --limit 10

# 仅评估检索指标
uv run python eval/eval_ragas.py --metrics retrieval

# 仅评估生成指标
uv run python eval/eval_ragas.py --metrics generation

# 安装依赖
uv sync
```

## 核心架构

```
src/
├── api.py            FastAPI 接口层：上传、检索、Agent 对话、状态查询、反馈
├── service.py        知识库服务层：文档上传、入库、向量/混合检索
├── agent.py          LangChain Agent：MCP SSE 客户端 + 动态工具加载 + 流式对话
├── mcp_server.py     MCP Server（SSE 传输，独立部署）：检索 + 数据库 MCP 工具
├── qa.py             RAG 问答：混合检索上下文 + DeepSeek 生成（单轮）
├── retriever.py      混合检索引擎：BM25 + 向量 + RRF + Cross-Encoder
├── loader.py         文档加载器：PyMuPDF (PDF) / DocxReader (DOCX) + 扫描版 PaddleOCR
├── chunker.py        中文分块：BGE token 计数 + SimilarityMergeNodeParser（保留作降级/参考）
├── markdown_chunker.py Markdown 语义分块器：原子解析 + SentenceSplitter + 相似度合并（三阶段流水线）
├── compat.py         ragas/langchain-community 兼容补丁（导入 ragas 前必须先 import）
├── settings.py       LlamaIndex Settings 全局依赖注入入口
├── database.py       SQLAlchemy 双引擎：同步(pymysql)供 grain_db，异步(aiomysql)供 chat_store
├── models.py         ORM 模型：Conversation / Message / Feedback / GrainMonitoring
├── chat_store.py     对话持久化：MySQL（conversations / messages / feedback 表）
├── session_cache.py  Redis 会话缓存：读缓存 + MySQL 兜底，异步不阻塞事件循环
├── grain_db.py       粮仓监测数据查询：grain_monitoring 表（供 MCP 调用）
├── cache_manager.py  语义缓存：FAISS 向量相似度匹配，减少重复 LLM 调用
├── prompt_loader.py  提示词/配置加载：从 prompts/ 读取，支持版本切换
├── state_manager.py  入库状态：SHA256 哈希增量，跳过未变更文件
├── ingest.py         一键入库 CLI 脚本
├── download.py       模型下载
├── static/           前端静态文件 (index.html)
├── bge-large-zh-v1.5/  本地 Embedding 模型（BGE-large-zh）
prompts/
├── system.txt        Agent 系统提示词
├── agent_config.json 模型配置（model / temperature / max_tokens）
chroma_data/          ChromaDB 向量库持久化目录
documents/            上传文档存储目录
temp_uploads/         上传临时文件目录（处理后自动清理）
```

## API 端点摘要

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 (index.html) |
| `/api/upload` | POST | 上传单个文档 → 自动入库 |
| `/api/upload-multiple` | POST | 批量上传文档 |
| `/api/search` | GET | 纯向量检索 |
| `/api/hybrid-search` | GET | 混合检索 (BM25+向量+精排) |
| `/api/ask` | GET | RAG 问答（检索+LLM 生成，单轮） |
| `/api/chat/stream` | POST | Agent 多轮对话（SSE 流式） |
| `/api/chat` | POST | Agent 多轮对话（非流式，备用） |
| `/api/conversations` | GET/POST | 对话列表 / 新建对话 |
| `/api/conversations/{cid}/messages` | GET | 获取对话消息历史 |
| `/api/conversations/{cid}` | DELETE | 删除对话 |
| `/api/feedback` | POST | 提交 👍/👎 反馈 |
| `/api/feedback/stats` | GET | 反馈统计 |
| `/api/status` | GET | 知识库状态（chunk 数/文件数等） |

## Agent 工具定义

Agent 通过 LangChain `@tool` 装饰器定义两个工具函数（`agent.py`），直接 Python 函数调用，不经过 MCP 协议：

| 工具 | 说明 |
|------|------|
| `search_knowledge_base(query, top_k=5)` | 混合检索知识库（BM25+向量+精排），返回文档片段列表 |
| `query_grain_data(hwdm, grain_name, start_date, end_date, production_area, limit=20, agg="none", group_by=None)` | 查询粮仓监测数据，支持筛选与聚合（avg/max/min/sum/count），可选按 hour/day/month 分组 |

## 关键设计要点

- **数据库双引擎**（`database.py`）：`sync_engine`（pymysql）供 `grain_db.py` 的 `@tool` 同步函数使用，`async_engine`（aiomysql）供 `chat_store.py` 的 FastAPI 异步上下文使用，避免同步阻塞事件循环
- **MCP 独立服务 + 动态工具加载**：`mcp_server.py` 作为独立进程运行（SSE 传输，默认 `localhost:8001`），暴露三个 MCP 工具：`search_kb`（向量检索）、`hybrid_search_kb`（混合检索）、`query_grain_data`（数据库查询）。`agent.py` 通过 `sse_client(MCP_SERVER_URL)` 连接，用 `langchain_mcp_adapters.load_mcp_tools()` 动态拉取工具列表（`get_agent()` 为 async）。Agent 不再直接 import 业务模块（`service` / `grain_db`），实现进程级解耦——主进程启动不再阻塞等 MCP 预热，MCP Server 可独立部署/扩缩/重启。启动方式：`uv run python src/mcp_server.py` 或 `MCP_HOST=0.0.0.0 MCP_PORT=8001 uv run python src/mcp_server.py`
- **双路召回 + 精排**：`retriever.py` 中 BM25 关键词 + 向量语义各召回 20 条，RRF 融合后取 15 条，再由 `BAAI/bge-reranker-v2-m3` Cross-Encoder 精排返回 top_k
- **SSE 流式事件格式**：`/api/chat/stream` 返回四种 SSE 事件类型 —— `event: thinking`（推理阶段文本）、`event: tool`（工具调用信息，格式 `工具名(参数)`）、`event: answer`（逐 token 推送回答）、`event: error`（异常信息）。前端通过 `[PHASE:ANSWER]` 标记区分思考过程和最终回答的存储边界
- **Agent 流式输出**：`agent.py` 使用 LangChain `create_agent` + `astream_events`（v2），工具调用前所有 LLM 输出为思考阶段，工具调用后为回答阶段。无工具调用的对话整个输出视为回答
- **语义缓存**：`cache_manager.py` 基于 FAISS IndexFlatIP 做向量相似度匹配（阈值 0.92），问题语义相似时直接返回缓存答案，避免重复 LLM 调用。仅缓存首轮对话、非涉库查询、长度 > 50 的回答。缓存满 500 条时淘汰最旧记录
- **Redis 热存储 + MySQL 异步持久化（`session_cache.py`）**：Redis 为主存储（读写主路径），MySQL 为冷持久化（异步批量刷入）。写路径：`RPUSH` 到 Redis LIST `msgs:{cid}:all` → debounce 3 秒 → 批量 `INSERT` MySQL。读路径：`LRANGE` 从 Redis LIST 取最新 N 条 → 命中返回；未命中从 MySQL 恢复 Redis。关闭时 `flush_all_pending()` 强制刷所有待持久化消息。Redis 不可用时静默降级为 MySQL 直读直写（等价于旧版行为）
- **单文件入库**：`service.py` 的 `ingest_file()` 会先删除该文件在 ChromaDB 中的旧 chunk，再重新分块写入，避免新旧共存。核心入库流水线为 2 阶段 `IngestionPipeline`：`MarkdownChunker(max_tokens=512, threshold=0.8)` → `Settings.embed_model`，策略为 `DocstoreStrategy.UPSERTS`
- **增量入库**：`state_manager.py` 基于文件 SHA256 哈希比对，跳过未变更文件；`upload_file()` 通过 `find_by_hash()` 也支持跨文件名去重
- **Markdown 语义分块**（`markdown_chunker.py`）：三阶段流水线配合 MinerU 结构化输出。阶段 1 `parse_atoms()`：正则扫描 Markdown，拆分为 heading / text / table / formula / list / image 语义原子，table/formula/image 标记为受保护。阶段 2 `_split_atoms()`：受保护原子原样保留，长文本原子走 SentenceSplitter 递归切分。阶段 3 `_similarity_merge()`：遍历节点对，token 预算（512）+ 边界余弦相似度（0.8）决定合并。章节路径（section_path）通过标题栈追踪，写入 chunk metadata。对纯文本文档（无 Markdown 结构）自动降级，全部归类为 text 原子走标准 SentenceSplitter + 相似度合并路径。旧方案 `chunker.py`（`SentenceSplitter` + `SimilarityMergeNodeParser` 分步调用、`chunk_documents()` 函数）保留作参考和降级备用
- **LlamaIndex Settings 依赖注入**（`settings.py`）：`configure_settings()` 在 `api.py` 启动时调用一次（幂等），全局注册 `Settings.embed_model`（BGE）、`Settings.llm`（DeepSeek via OpenAILike）、`Settings.node_parser`（SentenceSplitter, chunk_size=256）。其他模块通过 `Settings.xxx` 获取组件，不再手动跨模块传递。注意：`chunker.py` 的分块逻辑（BGE token 计数 / 512 token）是独立于 LlamaIndex node_parser 的自定义实现
- **embedding 模型本地加载**：`src/bge-large-zh-v1.5/`，离线可用，device=cpu。同时用于 embedding、BM25 分词和语义缓存的向量编码
- **扫描版 PDF OCR**（`loader.py`）：PDF 提取文本层后若内容少于 100 字符，自动 fallback 到 PaddleOCR 逐页识别（中文，200 DPI，置信度 > 0.5 保留），识别结果标记 `ocr=True`
- **粮仓监测数据库**：`grain_db.py` 查询 `agent_db.grain_monitoring` 表，支持按货位代码、粮种、日期、产地筛选。Agent 通过本地函数 `query_grain_data`（`@tool`）直接调用。与 `chat_store.py` 共用同一个 MySQL 库（`agent_db`）但操作不同的表
- **提示词外部管理**：`prompts/` 目录存放系统提示词（`system.txt`）和模型配置（`agent_config.json`）。`prompt_loader.py` 支持版本切换 —— `load_system_prompt("v2")` 会读取 `prompts/system_v2.txt`。`get_model_kwargs()` 中环境变量优先级高于配置文件
- **用户反馈闭环**：`api.py` 提供 `/api/feedback` 端点，通过 `chat_store.py` 的 `feedback` 表收集 👍/👎 评价。`ensure_feedback_table()` 在启动时和每次写入前自动建表

## 配置

### 环境变量（`.env` 文件，项目根目录）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | (必填) |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-chat` |
| `DEEPSEEK_TEMPERATURE` | 生成温度 | `0.3` |
| `DEEPSEEK_MAX_TOKENS` | 最大输出 token | `1024` |
| `REDIS_HOST` | Redis 地址 | `localhost` |
| `REDIS_PORT` | Redis 端口 | `6379` |
| `REDIS_DB` | Redis 数据库编号 | `0` |
| `REDIS_CACHE_TTL` | 缓存过期时间（秒） | `1800` |
| `REDIS_FLUSH_DEBOUNCE` | 消息刷 MySQL 的 debounce 秒数 | `3.0` |
| `MCP_SERVER_URL` | MCP 检索服务地址 | `http://localhost:8001/sse` |
| `MCP_HOST` | MCP Server 监听地址（仅 mcp_server 使用） | `0.0.0.0` |
| `MCP_PORT` | MCP Server 监听端口（仅 mcp_server 使用） | `8001` |

### 数据库

- MySQL `agent_db` 库：`chat_store.py`（conversations / messages / feedback）和 `grain_db.py`（grain_monitoring）共用
- `grain_monitoring` 表字段：粮温、仓湿、气湿、水分、脂肪酸值、杂质、不完善粒、容重、产地、算法分析结论等
- `feedback` 表在启动时自动创建（`ensure_feedback_table()`），conversations/messages 表需手动创建
- **注意**：`chat_store.py` 中 MySQL 凭据（`root` / 端口 3306）为硬编码，部署时需按实际环境修改

### 前置条件

- 项目根目录需创建 `.env` 文件，至少配置 `DEEPSEEK_API_KEY`（必填）
- 首次运行前需执行 `uv sync` 安装依赖
- BGE 模型目录 `src/bge-large-zh-v1.5/` 需存在（离线 embedding 模型）
- Cross-Encoder 精排模型 `bge-reranker-v2-m3` 需存在于 `~/.cache/huggingface/hub/` 或 `models/` 目录

### 前端

- 单一 HTML 文件 `src/static/index.html`，无构建步骤，通过 `/` 路由直接返回
- 使用 SSE（Server-Sent Events）与 `/api/chat/stream` 通信，展示思考过程 + 流式回答

### 评估模块

```
eval/
├── eval_ragas.py         Ragas 评估入口脚本（100 题全量/抽样）
├── ragas_evaluator.py    RagasEvaluator 封装类（LLM-as-a-Judge）
├── eval_retrieval.py     检索质量评估（MRR / NDCG / Hit@k）
├── eval_ttft.py          首字延迟评估（流式 vs 非流式）
├── eval_redis.py         Redis 缓存性能测试
├── eval_ragas_questions.json  100 道评估题目（含参考答案）
```

**重要**：任何导入 `ragas` 的代码必须先在模块最顶部 `import src.compat`，该补丁解决 `langchain_community.chat_models.vertexai.ChatVertexAI` 在 langchain-community ≥ 0.4.0 中被移除的兼容性问题。

### Python 依赖

`pyproject.toml` 使用清华 PyPI 镜像源，Python ≥ 3.13，主要依赖：FastAPI、LangChain、ChromaDB、llama-index 系列、sentence-transformers、PyMuPDF、jieba、rank-bm25、faiss、redis、pymysql、MCP、ragas

## 旧版代码

`智能体开发/` 目录是早期 Streamlit 原型（扫地机器人客服 demo），与当前 FastAPI 主项目无关，保留作参考。
