# 🌾 粮食仓储知识库 RAG 助手

基于文档检索增强生成（RAG）的智能问答系统，专注于粮食仓储领域。上传 PDF/DOCX 文档后，通过混合检索召回相关片段，由大模型生成准确回答。

## ✨ 功能特性

- **📄 多格式文档解析** — 支持 PDF（含扫描版 OCR）和 DOCX，Markdown 语义分块
- **🔍 混合检索引擎** — BM25 关键词 + 向量语义双路召回，RRF 融合，Cross-Encoder 精排
- **🤖 Agent 多轮对话** — 流式 SSE 输出，支持工具调用（知识库检索 + 粮仓数据库查询）
- **⚡ 语义缓存** — FAISS 向量相似度匹配，减少重复 LLM 调用
- **💾 会话持久化** — Redis 热存储 + MySQL 异步持久化，Redis 不可用时自动降级
- **📊 粮仓监测查询** — 支持按货位、粮种、日期、产地筛选，聚合统计（平均/最高/最低）
- **🛠️ MCP 解耦架构** — 检索服务独立进程，支持独立部署与扩缩
- **📈 评估体系** — 检索质量（MRR/NDCG/Hit@k）、TTFT 延迟、Ragas 语义评估

## 🚀 快速开始

### 环境要求

- Python ≥ 3.13
- MySQL（会话存储 + 粮仓监测数据）
- Redis（可选，用于会话热存储）

### 安装

```bash
# 克隆仓库
git clone https://github.com/ddx818/grain-knowledge-rag.git
cd grain-knowledge-rag

# 安装依赖（使用 uv）
uv sync

# 下载模型文件
uv run python src/download.py
```

### 配置

复制环境变量模板并填写配置：

```bash
cp .env.example .env
```

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | **必填** |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-chat` |
| `MCP_SERVER_URL` | MCP 检索服务地址 | `http://localhost:8001/sse` |

### 启动服务

```bash
# 1. 启动 MCP 检索服务（独立进程）
uv run python src/mcp_server.py

# 2. 启动 FastAPI 主服务（另一个终端）
uv run uvicorn src.api:app --reload --port 8000
```

访问 http://localhost:8000 进入 Web 对话界面。

### 文档入库

```bash
# 增量入库（跳过未变更文件）
uv run python src/ingest.py

# 全量重建（⚠️ 需先停止 API 服务）
uv run python src/ingest.py --full

# 测试模式（仅处理前 10 个文档）
uv run python src/ingest.py --test
```

## 📡 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 对话界面 |
| `/api/upload` | POST | 上传单个文档 |
| `/api/upload-multiple` | POST | 批量上传 |
| `/api/hybrid-search` | GET | 混合检索 |
| `/api/ask` | GET | RAG 问答（单轮） |
| `/api/chat/stream` | POST | Agent 多轮对话（SSE 流式） |
| `/api/chat` | POST | Agent 多轮对话（非流式） |
| `/api/conversations` | GET/POST | 对话列表管理 |
| `/api/feedback` | POST | 提交反馈 |
| `/api/status` | GET | 知识库状态 |

## 🧪 评估

```bash
# 检索质量评估（MRR / NDCG / Hit@k）
uv run python eval/eval_retrieval.py

# 首字延迟评估（流式 vs 非流式）
uv run python eval/eval_ttft.py

# Ragas 语义评估（LLM-as-a-Judge）
uv run python eval/eval_ragas.py --limit 10

# Redis 缓存性能测试
uv run python eval/eval_redis.py
```

## 🏗️ 架构

```
src/
├── api.py              FastAPI 接口层
├── service.py          知识库服务层
├── agent.py            LangChain Agent（MCP 客户端 + 流式）
├── mcp_server.py       MCP 检索服务（独立进程）
├── retriever.py        混合检索引擎（BM25 + 向量 + RRF + Cross-Encoder）
├── qa.py               RAG 问答
├── loader.py           文档加载器（PyMuPDF / DocxReader / PaddleOCR）
├── markdown_chunker.py Markdown 语义分块器（三阶段流水线）
├── cache_manager.py    FAISS 语义缓存
├── session_cache.py    Redis 会话缓存 + MySQL 持久化
├── chat_store.py       对话持久化（MySQL）
├── grain_db.py         粮仓监测数据查询
├── database.py         SQLAlchemy 双引擎（同步 + 异步）
├── prompt_loader.py    提示词加载
├── state_manager.py    入库状态管理
├── ingest.py           一键入库 CLI
├── download.py         模型下载脚本
└── static/index.html   前端页面
```

## 📄 许可证

[MIT License](LICENSE)
