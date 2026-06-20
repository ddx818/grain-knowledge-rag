"""MCP Server：将知识库检索能力暴露为 MCP 工具（SSE 传输，独立部署）。

启动方式：
    python src/mcp_server.py
    MCP_HOST=0.0.0.0 MCP_PORT=8001 uv run python src/mcp_server.py

端点：
    GET  /sse        SSE 长连接
    POST /messages/  客户端 JSON-RPC 请求
"""
import sys
from pathlib import Path
from contextlib import contextmanager

# 确保项目根目录在 sys.path 中（支持独立运行和 uv run）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 抑制 jieba 等库的 DEBUG 日志
import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from mcp.server.fastmcp import FastMCP
from src.service import KnowledgeBaseService


@contextmanager
def _silence_stdout():
    """临时将 sys.stdout 重定向到 stderr，防止 print/进度条破坏 MCP 通信协议。"""
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved


mcp = FastMCP("粮储知识库检索")

_kb = None


def _get_kb():
    global _kb
    if _kb is not None:
        return _kb
    _kb = KnowledgeBaseService()
    return _kb


@mcp.tool()
def search_kb(query: str, top_k: int = 5) -> list:
    """向量语义检索粮食仓储知识库，返回文档片段列表。适合按主题模糊搜索。"""
    with _silence_stdout():
        return _get_kb().search(query, top_k=top_k)


@mcp.tool()
def hybrid_search_kb(query: str, top_k: int = 5) -> list:
    """混合检索（BM25 关键词 + 向量语义 + Cross-Encoder 精排），
    返回更精准的文档片段列表。推荐优先使用此工具。"""
    with _silence_stdout():
        return _get_kb().hybrid_search(query, top_k=top_k)


@mcp.tool()
def query_grain_data(
    hwdm: str = "",
    grain_name: str = "",
    start_date: str = "",
    end_date: str = "",
    production_area: str = "",
    limit: int = 20,
    agg: str = "none",
    group_by: str = "",
) -> list:
    """查询粮仓实时监测数据库。可按货位代码、粮种名称、检测日期范围、产地筛选。
    支持两种模式：
    - 明细模式（agg="none"，默认）：返回原始监测记录，字段包括粮种、货位、
      检测时间、仓湿、气湿、粮温、仓外温度、最大/最小/平均粮温、水分、杂质、
      不完善粒、脂肪酸值、容重、产地、算法分析结论等
    - 聚合模式（agg="avg"|"max"|"min"|"sum"|"count"）：在数据库层完成统计计算，
      可选 group_by="hour"|"day"|"month" 按时间分组，返回聚合结果而非原始记录。
      聚合模式会为每个数值字段计算统计值并附加 record_count。

    参数说明：
    - hwdm: 货位代码（模糊匹配）
    - grain_name: 粮种名称（模糊匹配）
    - start_date: 检测起始日期，格式 YYYY-MM-DD
    - end_date: 检测截止日期，格式 YYYY-MM-DD
    - production_area: 产地（模糊匹配）
    - limit: 最多返回条数，默认 20
    - agg: 聚合函数，none（明细）/avg/max/min/sum/count
    - group_by: 聚合时的时间分组维度，hour/day/month（仅 agg≠none 时生效）
    """
    from src.grain_db import query_grain_data as do_query
    return do_query(
        hwdm=hwdm or None,
        grain_name=grain_name or None,
        start_date=start_date or None,
        end_date=end_date or None,
        production_area=production_area or None,
        limit=limit,
        agg=agg or "none",
        group_by=group_by or None,
    )


if __name__ == "__main__":
    import os

    # 预热：静默加载所有模型，避免首次工具调用时产生输出
    with _silence_stdout():
        kb = _get_kb()
        kb.embed_model                 # 加载 embedding 模型
        kb.hybrid_search("预热", top_k=1)  # 加载 BM25 + CrossEncoder
        print("MCP Server 预热完成，所有模型已加载", file=sys.stderr)

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))

    mcp.run(transport="sse", host=host, port=port)
