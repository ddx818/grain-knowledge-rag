"""
FastAPI 接口：文档上传、检索、Agent 对话、状态查询、反馈。

启动方式：
    python src/api.py
    或: uvicorn src.api:app --reload --port 8000
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from src.service import KnowledgeBaseService
from src.qa import QAService
from src import chat_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 Settings + 数据库表，后台异步连接 MCP Server。"""
    import logging
    import asyncio as _asyncio
    logger = logging.getLogger("uvicorn")

    from src.settings import configure_settings
    configure_settings()

    try:
        await chat_store.ensure_tables()
        logger.info("数据库表就绪")
    except Exception as e:
        logger.warning(f"数据库表检查失败: {e}")

    # 后台连接 MCP Server，不阻塞服务启动
    async def _connect_mcp():
        from src.agent import _ensure_mcp_session
        try:
            await _ensure_mcp_session()
            logger.info("MCP Server 连接就绪")
        except Exception as e:
            logger.warning(f"MCP Server 连接失败: {e}，将在首次请求时重试")

    _task = _asyncio.create_task(_connect_mcp())

    yield

    # 关闭前：取消 MCP 连接 + 刷所有待持久化消息到 MySQL
    _task.cancel()
    try:
        from src.agent import _close_mcp_session
        await _close_mcp_session()
    except Exception:
        pass

    try:
        from src.session_cache import flush_all_pending
        await flush_all_pending()
        logger.info("待持久化消息已全部刷入 MySQL")
    except Exception as e:
        logger.warning(f"关闭前刷消息失败: {e}")


app = FastAPI(title="RAG 知识库助手", lifespan=lifespan)

kb = KnowledgeBaseService()
qa = QAService(kb)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ================================================================
# 页面
# ================================================================

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ================================================================
# 上传
# ================================================================

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    tmp_path = PROJECT_ROOT / "temp_uploads" / (file.filename or "unknown")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    tmp_path.write_bytes(content)

    result = kb.upload_file(str(tmp_path))

    try:
        tmp_path.unlink()
    except Exception:
        pass

    if result["action"] in ("new", "update"):
        try:
            n = kb.ingest_file(result["dest"])
            result["chunks"] = n
        except Exception as e:
            result["chunks"] = 0
            result["ingest_error"] = str(e)

    return result


@app.post("/api/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        tmp_path = PROJECT_ROOT / "temp_uploads" / (file.filename or "unknown")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        tmp_path.write_bytes(content)

        result = kb.upload_file(str(tmp_path))

        try:
            tmp_path.unlink()
        except Exception:
            pass

        if result["action"] in ("new", "update"):
            try:
                n = kb.ingest_file(result["dest"])
                result["chunks"] = n
            except Exception as e:
                result["chunks"] = 0
                result["ingest_error"] = str(e)

        result["filename"] = file.filename
        results.append(result)

    return {"results": results}


# ================================================================
# 检索
# ================================================================

@app.get("/api/search")
async def search(q: str = Query(..., description="检索问题"), top_k: int = Query(5)):
    results = kb.search(q, top_k=top_k)
    return {"query": q, "results": results}


@app.get("/api/hybrid-search")
async def hybrid_search(q: str = Query(..., description="检索问题"), top_k: int = Query(5)):
    results = kb.hybrid_search(q, top_k=top_k)
    return {"query": q, "results": results}


@app.get("/api/ask")
async def ask(q: str = Query(..., description="用户问题"), top_k: int = Query(5)):
    result = qa.ask(q, top_k=top_k)
    return result


# ================================================================
# Agent 对话 + 对话管理
# ================================================================

@app.post("/api/chat/stream")
async def api_chat_stream(request: dict):
    from src.agent import chat_stream
    message = request.get("message", "")
    session_id = request.get("session_id", "")

    async def generate():
        async for token in chat_stream(message, session_id=session_id):
            yield token

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/chat")
async def api_chat(request: dict):
    from src.agent import chat
    message = request.get("message", "")
    session_id = request.get("session_id", "")
    answer = await chat(message, session_id=session_id)
    return {"answer": answer, "session_id": session_id}


@app.post("/api/conversations")
async def create_conv():
    from src.session_cache import create_conversation_async, invalidate_conversation_list
    conv = await create_conversation_async()
    invalidate_conversation_list()
    return conv


@app.get("/api/conversations")
async def list_conv():
    from src.session_cache import get_conversation_list_cached
    return await get_conversation_list_cached()


@app.get("/api/conversations/{cid}/messages")
async def get_conv_messages(cid: str):
    from src.session_cache import get_conversation_cached, get_messages_cached
    conv = await get_conversation_cached(cid)
    if not conv:
        return JSONResponse({"error": "not found"}, 404)
    return {"conversation": conv, "messages": await get_messages_cached(cid)}


@app.delete("/api/conversations/{cid}")
async def delete_conv(cid: str):
    from src.agent import clear_history
    await clear_history(cid)
    return {"status": "ok"}


# ================================================================
# 用户反馈
# ================================================================

@app.post("/api/feedback")
async def submit_feedback(request: dict):
    cid = request.get("conversation_id", "")
    rating = request.get("rating", "")
    comment = request.get("comment", "")

    if not cid or rating not in ("positive", "negative"):
        return JSONResponse({"error": "缺少 conversation_id 或 rating 无效"}, 400)

    # ── 拉取对话上下文（供人工排查，直接存入 feedback 记录） ──
    import json as _json
    msgs = await chat_store.get_messages(cid)
    context = _json.dumps(msgs, ensure_ascii=False) if msgs else ""

    fid = await chat_store.add_feedback(cid, rating, comment, context=context)

    # ── 监控埋点：记录反馈事件（关联到该对话使用的工具） ──
    import asyncio as _asyncio
    from src.monitoring import get_metrics
    metrics = get_metrics()
    await metrics.feedback_received(rating, session_id=cid)

    # ── 👎 反馈：异步失效语义缓存 ──
    if rating == "negative":
        async def _invalidate():
            try:
                if msgs and msgs[0]["role"] == "user":
                    from src.cache_manager import get_cache
                    removed = get_cache().remove(msgs[0]["content"])
                    if removed > 0:
                        import logging
                        logging.getLogger("uvicorn").info(
                            f"feedback 👎 → 缓存已失效 {removed} 条 (conv={cid})")
            except Exception:
                pass
        _asyncio.create_task(_invalidate())

    return {"status": "ok", "id": fid}


@app.get("/api/feedback/stats")
async def feedback_stats():
    return await chat_store.get_feedback_stats()


# ================================================================
# 状态
# ================================================================

@app.get("/api/status")
async def status():
    return kb.get_status()


# ================================================================
# 运行时监控
# ================================================================

@app.get("/api/metrics")
async def metrics():
    """返回运行时监控指标快照。

    包含三个核心维度：
    - tools: 各工具调用次数、成功率、空结果率、P50/P95/P99 延迟
    - sse:   流式连接开始/完成/异常中断计数与比率
    - uptime_seconds: 服务运行时长
    """
    from src.monitoring import get_metrics
    return await get_metrics().snapshot()


@app.post("/api/metrics/reset")
async def metrics_reset():
    """重置所有运行时指标计数器（用于排障后清零）。"""
    from src.monitoring import get_metrics
    await get_metrics().reset()
    return {"status": "ok"}


# ================================================================
# 启动
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
