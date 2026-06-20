"""
运行时监控指标收集器（单例，async 安全）。

三个核心指标：
1. 工具调用成功率 + 延迟分位数 — 按工具名分组，P50/P95/P99
2. SSE 流完整性 — 连接开始/完成/异常断开计数
3. RAG 端到端回答有效率 — 工具调用结果非空率 + 反馈交叉关联

所有指标存内存，通过 /api/metrics 暴露快照，重启清零。
"""

import time
import asyncio
import threading
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from langchain.agents.middleware.types import AgentMiddleware

_MAX_LATENCIES = 1000


@dataclass
class ToolMetrics:
    total_calls: int = 0
    success_calls: int = 0
    error_calls: int = 0
    empty_results: int = 0
    latencies: list[float] = field(default_factory=list)
    last_error: Optional[str] = None
    last_call_time: float = 0.0


@dataclass
class ModelMetrics:
    """大模型调用指标。"""
    total_calls: int = 0
    success_calls: int = 0
    error_calls: int = 0
    latencies: list[float] = field(default_factory=list)
    last_error: Optional[str] = None
    last_call_time: float = 0.0


@dataclass
class SSEMetrics:
    streams_started: int = 0
    streams_completed: int = 0
    streams_errored: int = 0
    streams_with_tool: int = 0
    streams_without_tool: int = 0


@dataclass
class FeedbackMetrics:
    positive: int = 0
    negative: int = 0
    # 按工具名分组的 👎 计数（关联到具体工具调用）
    negative_by_tool: dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """运行时监控指标收集器（单例）。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._tool_metrics: dict[str, ToolMetrics] = defaultdict(ToolMetrics)
        self._model = ModelMetrics()
        self._sse = SSEMetrics()
        self._feedback = FeedbackMetrics()
        self._start_time = time.time()
        self._pending_tools: dict[str, tuple[str, float]] = {}
        # 记录最近的 SSE 流涉及的 conversation_id → 工具名列表（供反馈关联）
        self._recent_tools: dict[str, list[str]] = {}

    # ── SSE 流生命周期 ──

    async def stream_started(self):
        async with self._lock:
            self._sse.streams_started += 1

    async def stream_completed(self, *, had_tool: bool = False):
        async with self._lock:
            self._sse.streams_completed += 1
            if had_tool:
                self._sse.streams_with_tool += 1
            else:
                self._sse.streams_without_tool += 1

    async def stream_errored(self, *, had_tool: bool = False):
        async with self._lock:
            self._sse.streams_errored += 1
            if had_tool:
                self._sse.streams_with_tool += 1

    # ── 工具调用生命周期 ──

    async def tool_started(self, run_id: str, tool_name: str):
        async with self._lock:
            self._pending_tools[run_id] = (tool_name, time.time())

    async def tool_completed(self, run_id: str, *, result_empty: bool = False):
        async with self._lock:
            entry = self._pending_tools.pop(run_id, None)
            if entry is None:
                return
            tool_name, t0 = entry
            latency = (time.time() - t0) * 1000
            tm = self._tool_metrics[tool_name]
            tm.total_calls += 1
            tm.success_calls += 1
            tm.last_call_time = time.time()
            if result_empty:
                tm.empty_results += 1
            if len(tm.latencies) < _MAX_LATENCIES:
                tm.latencies.append(latency)

    async def tool_errored(self, run_id: str, error: str):
        async with self._lock:
            entry = self._pending_tools.pop(run_id, None)
            if entry is None:
                return
            tool_name, t0 = entry
            latency = (time.time() - t0) * 1000
            tm = self._tool_metrics[tool_name]
            tm.total_calls += 1
            tm.error_calls += 1
            tm.last_error = error
            tm.last_call_time = time.time()
            if len(tm.latencies) < _MAX_LATENCIES:
                tm.latencies.append(latency)

    async def flush_pending(self):
        """将未完成的工具调用全部标记为 errored（用于异常恢复）。"""
        async with self._lock:
            for run_id in list(self._pending_tools.keys()):
                await self.tool_errored(run_id, "stream aborted before tool_end")

    # ── 同步安全方法（供中间件 wrap_tool_call 使用） ──

    def record_tool_success(self, tool_name: str, latency_ms: float,
                            result_empty: bool = False):
        """同步记录工具调用成功。"""
        with self._sync_lock:
            tm = self._tool_metrics[tool_name]
            tm.total_calls += 1
            tm.success_calls += 1
            tm.last_call_time = time.time()
            if result_empty:
                tm.empty_results += 1
            if len(tm.latencies) < _MAX_LATENCIES:
                tm.latencies.append(latency_ms)

    def record_tool_error(self, tool_name: str, latency_ms: float, error: str):
        """同步记录工具调用失败。"""
        with self._sync_lock:
            tm = self._tool_metrics[tool_name]
            tm.total_calls += 1
            tm.error_calls += 1
            tm.last_error = error
            tm.last_call_time = time.time()
            if len(tm.latencies) < _MAX_LATENCIES:
                tm.latencies.append(latency_ms)

    # ── 模型调用记录（同步安全，供 wrap_model_call 使用） ──

    def record_model_success(self, latency_ms: float):
        """同步记录大模型调用成功。"""
        with self._sync_lock:
            self._model.total_calls += 1
            self._model.success_calls += 1
            self._model.last_call_time = time.time()
            if len(self._model.latencies) < _MAX_LATENCIES:
                self._model.latencies.append(latency_ms)

    def record_model_error(self, latency_ms: float, error: str):
        """同步记录大模型调用失败。"""
        with self._sync_lock:
            self._model.total_calls += 1
            self._model.error_calls += 1
            self._model.last_error = error
            self._model.last_call_time = time.time()
            if len(self._model.latencies) < _MAX_LATENCIES:
                self._model.latencies.append(latency_ms)

    # ── 反馈 ──

    async def record_conv_tools(self, session_id: str, tool_names: list[str]):
        """记录某个 SSE 流涉及的工具名（供反馈交叉关联）。最多保留 200 条。"""
        async with self._lock:
            if len(self._recent_tools) >= 200:
                # 淘汰最旧的一条
                oldest = next(iter(self._recent_tools))
                del self._recent_tools[oldest]
            self._recent_tools[session_id] = tool_names

    async def feedback_received(self, rating: str, session_id: str = ""):
        """记录用户反馈，并与涉及的工具调用交叉关联。"""
        async with self._lock:
            if rating == "positive":
                self._feedback.positive += 1
            elif rating == "negative":
                self._feedback.negative += 1
                tools = self._recent_tools.get(session_id, [])
                for t in tools:
                    self._feedback.negative_by_tool[t] = \
                        self._feedback.negative_by_tool.get(t, 0) + 1

    # ── 快照 ──

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sv = sorted(values)
        idx = int(len(sv) * p / 100)
        return sv[min(idx, len(sv) - 1)]

    async def snapshot(self) -> dict:
        """返回当前指标快照。"""
        # 先读工具和模型指标（由 _sync_lock 保护，中间件同步写入）
        with self._sync_lock:
            tools = {}
            for name, tm in self._tool_metrics.items():
                total = tm.total_calls
                success = tm.success_calls
                tools[name] = {
                    "total_calls": total,
                    "success_calls": success,
                    "error_calls": tm.error_calls,
                    "empty_results": tm.empty_results,
                    "error_rate": round(tm.error_calls / total, 4) if total > 0 else 0.0,
                    "empty_rate": round(tm.empty_results / success, 4) if success > 0 else 0.0,
                    "latency_p50_ms": round(self._percentile(tm.latencies, 50), 1),
                    "latency_p95_ms": round(self._percentile(tm.latencies, 95), 1),
                    "latency_p99_ms": round(self._percentile(tm.latencies, 99), 1),
                    "last_error": tm.last_error,
                }

            total_m = self._model.total_calls
            model = {
                "total_calls": self._model.total_calls,
                "success_calls": self._model.success_calls,
                "error_calls": self._model.error_calls,
                "error_rate": round(self._model.error_calls / total_m, 4) if total_m > 0 else 0.0,
                "latency_p50_ms": round(self._percentile(self._model.latencies, 50), 1),
                "latency_p95_ms": round(self._percentile(self._model.latencies, 95), 1),
                "latency_p99_ms": round(self._percentile(self._model.latencies, 99), 1),
                "last_error": self._model.last_error,
            }

        # 再读 SSE 和反馈指标（由 _lock 保护，异步写入）
        async with self._lock:
            total_s = self._sse.streams_started
            sse = {
                "streams_started": self._sse.streams_started,
                "streams_completed": self._sse.streams_completed,
                "streams_errored": self._sse.streams_errored,
                "streams_with_tool": self._sse.streams_with_tool,
                "streams_without_tool": self._sse.streams_without_tool,
                "error_rate": round(self._sse.streams_errored / total_s, 4) if total_s > 0 else 0.0,
                "completion_rate": round(self._sse.streams_completed / total_s, 4) if total_s > 0 else 0.0,
            }

            total_fb = self._feedback.positive + self._feedback.negative
            feedback = {
                "positive": self._feedback.positive,
                "negative": self._feedback.negative,
                "total": total_fb,
                "negative_rate": round(self._feedback.negative / total_fb, 4) if total_fb > 0 else 0.0,
                "negative_by_tool": dict(self._feedback.negative_by_tool),
            }

            return {
                "uptime_seconds": round(time.time() - self._start_time, 0),
                "sse": sse,
                "model": model,
                "tools": tools,
                "feedback": feedback,
            }

    async def reset(self):
        """重置所有计数器（用于测试/排障后清零）。"""
        async with self._lock:
            self._tool_metrics.clear()
            self._model = ModelMetrics()
            self._sse = SSEMetrics()
            self._feedback = FeedbackMetrics()
            self._start_time = time.time()
            self._pending_tools.clear()
            self._recent_tools.clear()


# ── 全局单例 ──

_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


# ═══════════════════════════════════════════════════════════════
# MonitoringMiddleware — 用 Agent 中间件替代事件流中的工具埋点
# ═══════════════════════════════════════════════════════════════

def _check_tool_result_empty(result) -> bool:
    """判断 ToolMessage 的工具返回是否为空。"""
    if result is None:
        return True
    content = getattr(result, "content", None)
    if content is None:
        return True
    if isinstance(content, str):
        stripped = content.strip()
        return stripped == "" or stripped == "[]"
    if isinstance(content, list):
        return len(content) == 0
    return False


class MonitoringMiddleware(AgentMiddleware):
    """Agent 中间件：拦截工具调用，记录延迟、空结果、错误。

    替代原先 agent.py 中通过 astream_events 事件回调手写的
    tool_started / tool_completed / flush_pending 逻辑。

    用法：
        agent = create_agent(model, tools=[...], middleware=[MonitoringMiddleware()])
    """

    def __init__(self):
        super().__init__()
        self._metrics = get_metrics()
        self._tools_called: list[str] = []

    @property
    def name(self) -> str:
        return "MonitoringMiddleware"

    def get_and_clear_tools(self) -> list[str]:
        """返回本轮调用的工具名列表并清空（供反馈交叉关联）。"""
        names = list(self._tools_called)
        self._tools_called.clear()
        return names

    # ── 同步版本（LangGraph 同步执行路径） ──

    def wrap_tool_call(self, request, handler):
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        tool_name = request.tool_call.get("name", "unknown")
        self._tools_called.append(tool_name)
        t0 = time.time()

        try:
            result = handler(request)
            latency = (time.time() - t0) * 1000
            result_empty = _check_tool_result_empty(result)
            self._metrics.record_tool_success(tool_name, latency,
                                              result_empty=result_empty)
            return result
        except Exception as e:
            latency = (time.time() - t0) * 1000
            self._metrics.record_tool_error(tool_name, latency, str(e))
            raise

    # ── 异步版本（我们使用的路径） ──

    async def awrap_tool_call(self, request, handler):
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        tool_name = request.tool_call.get("name", "unknown")
        self._tools_called.append(tool_name)
        t0 = time.time()

        try:
            result = await handler(request)
            latency = (time.time() - t0) * 1000
            result_empty = _check_tool_result_empty(result)
            self._metrics.record_tool_success(tool_name, latency,
                                              result_empty=result_empty)
            return result
        except Exception as e:
            latency = (time.time() - t0) * 1000
            self._metrics.record_tool_error(tool_name, latency, str(e))
            raise

    # ── 大模型调用拦截 ──

    def wrap_model_call(self, request, handler):
        """包裹大模型调用，记录延迟和异常。"""
        t0 = time.time()
        try:
            result = handler(request)
            latency = (time.time() - t0) * 1000
            self._metrics.record_model_success(latency)
            return result
        except Exception as e:
            latency = (time.time() - t0) * 1000
            self._metrics.record_model_error(latency, str(e))
            raise

    async def awrap_model_call(self, request, handler):
        """异步包裹大模型调用，记录延迟和异常。"""
        t0 = time.time()
        try:
            result = await handler(request)
            latency = (time.time() - t0) * 1000
            self._metrics.record_model_success(latency)
            return result
        except Exception as e:
            latency = (time.time() - t0) * 1000
            self._metrics.record_model_error(latency, str(e))
            raise
