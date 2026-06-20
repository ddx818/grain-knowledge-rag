"""
LangChain Agent：思考过程可见 + token 级流式输出 + 多轮对话。

使用 create_agent + astream_events 实现真正的逐 token 流式输出：
- on_chat_model_stream → 逐 token 推送（打字机效果）
- on_tool_start → 展示工具调用信息
- on_tool_end → 标记思考/回答阶段切换

工具通过 MCP 协议从独立部署的 mcp_server 动态加载（SSE 传输）。
Agent 不直接依赖业务模块（service / grain_db），实现进程级解耦。

流式输出格式：
  思考内容 + [调用工具: xxx] + [PHASE:ANSWER] + 回答内容
前端通过 [PHASE:ANSWER] 标记区分思考过程和最终回答。
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools

from mcp import ClientSession
from mcp.client.sse import sse_client

from src.monitoring import get_metrics, MonitoringMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")

# ---- MCP 会话管理（SSE 传输，连接独立部署的 mcp_server） ----

_mcp_session: ClientSession | None = None
_session_ctx = None


async def _ensure_mcp_session():
    """建立到独立 MCP Server 的 SSE 连接（单例，复用，自动重连）。"""
    global _mcp_session, _session_ctx
    if _mcp_session is not None:
        try:
            # 快速探活：列出工具验证连接是否仍有效
            await _mcp_session.list_tools()
            return _mcp_session
        except Exception:
            _mcp_session = None
            _session_ctx = None

    _session_ctx = sse_client(MCP_SERVER_URL)
    read, write = await _session_ctx.__aenter__()

    session = ClientSession(read, write)
    _mcp_session = await session.__aenter__()
    await _mcp_session.initialize()

    return _mcp_session


async def _close_mcp_session():
    """关闭 MCP 连接（应用退出时调用）。"""
    global _mcp_session, _session_ctx
    if _session_ctx:
        await _session_ctx.__aexit__(None, None, None)
    _mcp_session = None
    _session_ctx = None


# ================================================================
# 工具加载（从 MCP Server 动态拉取）
# ================================================================

_mcp_tools_cache = None


async def _get_mcp_tools():
    """从 MCP Server 动态拉取工具列表，转为 LangChain 工具。"""
    global _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    session = await _ensure_mcp_session()
    _mcp_tools_cache = await load_mcp_tools(session)
    return _mcp_tools_cache


# ================================================================
# Agent 构建（单例）
# ================================================================

_agent = None
_monitoring_mw = MonitoringMiddleware()


async def get_agent():
    """获取 Agent 实例（单例）。工具列表从 MCP Server 动态加载。"""
    global _agent
    if _agent is not None:
        return _agent

    from src.prompt_loader import get_model_kwargs, load_system_prompt

    kwargs = get_model_kwargs()
    kwargs["streaming"] = True

    llm = ChatOpenAI(**kwargs)
    tools = await _get_mcp_tools()

    _agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=load_system_prompt(),
        middleware=[_monitoring_mw],
    )

    return _agent


# ================================================================
# 对话接口
# ================================================================

def _sse(event: str, data: str) -> str:
    """构造一条 SSE 消息。"""
    return f"event: {event}\ndata: {data}\n\n"


async def chat_stream(message: str, session_id: str = "default"):
    """
    流式多轮对话，SSE 协议推送。

    事件类型：
      event: thinking  — 思考阶段文本
      event: tool      — 工具调用
      event: answer    — 回答 token（逐 token 推送）
      event: error     — 异常
    """
    from src.session_cache import (
        get_messages_cached,
        add_message_async, create_conversation_async, get_conversation_async,
        update_title_async,
    )

    if not await get_conversation_async(session_id):
        await create_conversation_async(title="新对话", cid=session_id)

    # 加载历史（走 Redis 缓存）
    past = await get_messages_cached(session_id, limit=10)
    messages = []
    for m in past:
        role = "user" if m["role"] == "user" else "assistant"
        messages.append({"role": role, "content": m["content"]})

    # 保存用户消息
    await add_message_async(session_id, "user", message)
    messages.append({"role": "user", "content": message})
    # 消息已通过 RPUSH 写入 Redis LIST，无需手动更新缓存

    # 自动标题
    if len(past) == 0:
        title = message[:20] + ("..." if len(message) > 20 else "")
        await update_title_async(session_id, title)

    # ── 语义缓存检查 ──
    from src.cache_manager import get_cache
    cache = get_cache()
    if len(past) == 0:
        cached = cache.search(message)
        if cached:
            # ── 监控：缓存命中也算一次完整的无工具流 ──
            metrics_early = get_metrics()
            await metrics_early.stream_started()
            await metrics_early.stream_completed(had_tool=False)
            yield _sse("answer", cached)
            await add_message_async(session_id, "assistant",
                                    "[PHASE:ANSWER]\n" + cached)
            messages.append({"role": "assistant",
                           "content": "[PHASE:ANSWER]\n" + cached})
            return

    agent = await get_agent()
    buf = []                   # 阶段未确定前的 token 缓冲
    thinking_parts = []        # 思考阶段文本片段
    answer_parts = []          # 回答阶段文本片段
    phase_decided = False      # 是否已发生工具调用
    thinking_flushed = False   # 是否已推送 thinking 事件
    used_db_tool = False       # 是否调用了数据库工具（涉库不缓存）

    # ── 监控埋点 ──
    metrics = get_metrics()
    await metrics.stream_started()
    had_tool = False
    stream_errored_flag = False

    try:
        event_stream = agent.astream_events(
            {"messages": messages},
            version="v2",
        )
        async for event in event_stream:
            kind = event["event"]

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if not content:
                    continue

                if thinking_flushed:
                    answer_parts.append(content)
                    yield _sse("answer", content)
                else:
                    buf.append(content)

            elif kind == "on_tool_start":
                if not phase_decided:
                    thinking_parts = buf
                    buf = []
                    thinking_text = "".join(thinking_parts)
                    if thinking_text.strip():
                        yield _sse("thinking", thinking_text)
                        thinking_flushed = True
                    phase_decided = True

                had_tool = True
                tool_name = event.get("name", "")
                if tool_name == "query_grain_data":
                    used_db_tool = True

                tool_input = event.get("data", {}).get("input", {})
                args_str = str(tool_input.get("query", ""))
                tool_info = f"{tool_name}({args_str})"
                thinking_parts.append(tool_info)
                yield _sse("tool", tool_info)

            elif kind == "on_tool_end":
                if not thinking_flushed:
                    # 思考缓冲中的 token（工具调用之前的 LLM 输出）
                    # — 已在 on_tool_start 中推送为 thinking 事件
                    thinking_flushed = True
                    # 切换为 answer 阶段后，先把 on_tool_start/end 之间
                    # 缓冲的 token 推出去
                    for t in buf:
                        answer_parts.append(t)
                        yield _sse("answer", t)
                    buf = []
                phase_decided = True

    except Exception as e:
        stream_errored_flag = True
        err_text = f"检索服务异常：{e}"
        if not thinking_flushed:
            thinking_flushed = True
        answer_parts.append(err_text)
        yield _sse("error", err_text)

    # 无工具调用 → 整个输出都是回答
    if not phase_decided:
        answer_parts = buf
        yield _sse("answer", "".join(buf))
    elif buf:
        answer_parts.extend(buf)
        yield _sse("answer", "".join(buf))

    # ── 记录 SSE 流结果 ──
    if stream_errored_flag:
        await metrics.stream_errored(had_tool=had_tool)
    else:
        await metrics.stream_completed(had_tool=had_tool)
    conv_tools = _monitoring_mw.get_and_clear_tools()
    if conv_tools:
        await metrics.record_conv_tools(session_id, conv_tools)

    # 保存到 MySQL
    thinking_text = "".join(thinking_parts).strip()
    answer_text = "".join(answer_parts).strip()

    if thinking_text and answer_text:
        full_text = thinking_text + "\n[PHASE:ANSWER]\n" + answer_text
    elif thinking_text:
        full_text = thinking_text
    else:
        full_text = answer_text

    # ── 存入语义缓存（不涉及数据库查询的回答才缓存） ──
    if answer_text and len(answer_text) > 50 and not used_db_tool:
        cache.add(message, answer_text)

    await add_message_async(session_id, "assistant", full_text)
    messages.append({"role": "assistant", "content": full_text})
    # 消息已通过 RPUSH 写入 Redis LIST，无需手动更新缓存


async def chat(message: str, session_id: str) -> str:
    """多轮对话（非流式，备用）。"""
    from src.session_cache import (
        get_messages_cached,
        add_message_async, create_conversation_async, get_conversation_async,
        update_title_async,
    )

    if not await get_conversation_async(session_id):
        await create_conversation_async(title="新对话", cid=session_id)

    past = await get_messages_cached(session_id, limit=10)
    messages = []
    for m in past:
        role = "user" if m["role"] == "user" else "assistant"
        messages.append({"role": role, "content": m["content"]})

    await add_message_async(session_id, "user", message)
    messages.append({"role": "user", "content": message})
    # 消息已通过 RPUSH 写入 Redis LIST，无需手动更新缓存

    if len(past) == 0:
        title = message[:20] + ("..." if len(message) > 20 else "")
        await update_title_async(session_id, title)

    agent = await get_agent()
    # ── 监控 ──
    metrics_chat = get_metrics()
    await metrics_chat.stream_started()
    result = agent.invoke({"messages": messages})
    conv_tools_chat = _monitoring_mw.get_and_clear_tools()
    await metrics_chat.stream_completed(had_tool=len(conv_tools_chat) > 0)
    if conv_tools_chat:
        await metrics_chat.record_conv_tools(session_id, conv_tools_chat)

    all_msgs = result["messages"]
    thinking_parts = []
    answer_parts = []
    seen_tool = False

    for m in all_msgs:
        if m.type == "tool":
            seen_tool = True
            continue
        if m.type != "ai":
            continue
        content = m.content if hasattr(m, "content") else ""
        if not isinstance(content, str) or not content.strip():
            continue
        has_tc = hasattr(m, "tool_calls") and m.tool_calls
        if has_tc:
            thinking_parts.append(content)
        elif seen_tool:
            answer_parts.append(content)
        else:
            thinking_parts.append(content)

    if not seen_tool:
        answer_parts = thinking_parts
        thinking_parts = []

    thinking_text = "\n".join(thinking_parts).strip()
    answer_text = "\n".join(answer_parts).strip()

    if thinking_text and answer_text:
        full_text = thinking_text + "\n[PHASE:ANSWER]\n" + answer_text
    elif thinking_text:
        full_text = thinking_text
    else:
        full_text = answer_text

    await add_message_async(session_id, "assistant", full_text)
    messages.append({"role": "assistant", "content": full_text})
    # 消息已通过 RPUSH 写入 Redis LIST，无需手动更新缓存
    return answer_text or full_text


async def clear_history(session_id: str):
    from src.session_cache import (
        invalidate_messages, invalidate_conversation, invalidate_conversation_list,
        delete_conversation_async,
    )
    await delete_conversation_async(session_id)
    invalidate_messages(session_id)
    invalidate_conversation(session_id)
    invalidate_conversation_list()
