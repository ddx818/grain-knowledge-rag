"""
粮仓实时监测数据查询模块。
供 MCP Server / Agent 工具调用，使用 SQLAlchemy 同步 ORM。

支持两种模式：
- 明细模式（agg="none"，默认）：返回原始监测记录
- 聚合模式（agg=avg/max/min/sum/count）：在 SQL 层完成计算，
  可选 group_by=hour/day/month 进行时间维度分组。
"""

from typing import Optional, Dict, Any

from sqlalchemy import select, desc, func
from sqlalchemy.sql import Select

from src.database import SyncSessionLocal
from src.models import GrainMonitoring

# ── 聚合函数映射 ──
_AGG_FUNCS = {
    "avg": func.avg,
    "max": func.max,
    "min": func.min,
    "sum": func.sum,
    "count": func.count,
}

# ── 可聚合的数值列 ──
_AGG_COLUMNS: dict[str, Any] = {
    "avg_temper": GrainMonitoring.avg_temper,
    "max_temper": GrainMonitoring.max_temper,
    "min_temper": GrainMonitoring.min_temper,
    "inner_temper": GrainMonitoring.inner_temper,
    "outer_temper": GrainMonitoring.outer_temper,
    "inner_humidity": GrainMonitoring.inner_humidity,
    "outer_humidity": GrainMonitoring.outer_humidity,
    "moisture_content": GrainMonitoring.moisture_content,
    "impurity_ratio": GrainMonitoring.impurity_ratio,
    "imperfect_grain": GrainMonitoring.imperfect_grain,
    "fatty_acid_ester": GrainMonitoring.fatty_acid_ester,
    "unit_weight": GrainMonitoring.unit_weight,
}


def _build_where(stmt: Select, hwdm, grain_name, start_date, end_date, production_area):
    """在已有 stmt 上叠加 WHERE 条件，返回新 stmt。"""
    if hwdm:
        stmt = stmt.where(GrainMonitoring.hwdm.like(f"%{hwdm}%"))
    if grain_name:
        stmt = stmt.where(GrainMonitoring.grain_name.like(f"%{grain_name}%"))
    if start_date:
        stmt = stmt.where(GrainMonitoring.check_date >= start_date)
    if end_date:
        stmt = stmt.where(GrainMonitoring.check_date <= end_date)
    if production_area:
        stmt = stmt.where(GrainMonitoring.production_area.like(f"%{production_area}%"))
    return stmt


def _query_detail(
    hwdm, grain_name, start_date, end_date, production_area, limit
) -> list[Dict[str, Any]]:
    """明细模式：返回逐条原始记录。"""
    stmt = select(GrainMonitoring)
    stmt = _build_where(stmt, hwdm, grain_name, start_date, end_date, production_area)
    stmt = stmt.order_by(
        desc(GrainMonitoring.check_date), desc(GrainMonitoring.check_time)
    ).limit(limit)

    with SyncSessionLocal() as session:
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "hwdm": r.hwdm,
                "grain_name": r.grain_name,
                "check_date": str(r.check_date) if r.check_date else None,
                "check_time": str(r.check_time) if r.check_time else None,
                "inner_humidity": r.inner_humidity,
                "outer_humidity": r.outer_humidity,
                "inner_temper": r.inner_temper,
                "outer_temper": r.outer_temper,
                "max_temper": r.max_temper,
                "min_temper": r.min_temper,
                "avg_temper": r.avg_temper,
                "moisture_content": r.moisture_content,
                "impurity_ratio": r.impurity_ratio,
                "imperfect_grain": r.imperfect_grain,
                "fatty_acid_ester": r.fatty_acid_ester,
                "unit_weight": r.unit_weight,
                "production_area": r.production_area,
                "algorithm_analysis_conclusion": r.algorithm_analysis_conclusion,
            }
            for r in rows
        ]


def _query_agg(
    agg, group_by,
    hwdm, grain_name, start_date, end_date, production_area, limit,
) -> list[Dict[str, Any]]:
    """聚合模式：在 SQL 层完成 AVG/MAX/MIN/SUM/COUNT，可选时间分组。"""
    agg_fn = _AGG_FUNCS[agg]

    # ── 构建 SELECT 列 ──
    select_cols = []

    # 时间分组列
    time_label = None
    if group_by == "hour":
        # MySQL: DATE_FORMAT 或 concat + left
        time_expr = func.concat(
            GrainMonitoring.check_date, " ",
            func.left(GrainMonitoring.check_time, 2)
        ).label("time_group")
        select_cols.append(time_expr)
        time_label = "time_group"
    elif group_by == "day":
        time_expr = GrainMonitoring.check_date.label("time_group")
        select_cols.append(time_expr)
        time_label = "time_group"
    elif group_by == "month":
        time_expr = func.left(GrainMonitoring.check_date, 7).label("time_group")
        select_cols.append(time_expr)
        time_label = "time_group"

    # 聚合列
    agg_prefix = agg
    for col_name, col in _AGG_COLUMNS.items():
        select_cols.append(agg_fn(col).label(f"{agg_prefix}_{col_name}"))

    # 记录数
    select_cols.append(func.count().label("record_count"))

    # ── 构建查询 ──
    stmt = select(*select_cols)
    stmt = _build_where(stmt, hwdm, grain_name, start_date, end_date, production_area)

    # GROUP BY
    if group_by == "hour":
        stmt = stmt.group_by(
            GrainMonitoring.check_date,
            func.left(GrainMonitoring.check_time, 2)
        )
    elif group_by == "day":
        stmt = stmt.group_by(GrainMonitoring.check_date)
    elif group_by == "month":
        stmt = stmt.group_by(func.left(GrainMonitoring.check_date, 7))

    # 排序与限制
    if time_label:
        stmt = stmt.order_by(desc(time_label)).limit(limit)
    else:
        stmt = stmt.limit(1)

    # ── 执行 ──
    with SyncSessionLocal() as session:
        rows = session.execute(stmt).all()
        results = []
        for row in rows:
            d = dict(row._mapping)
            # 清理数值：None 保留，float 保留原生精度
            d["record_count"] = int(d.get("record_count", 0))
            results.append(d)
        return results


# ── 公开接口 ──

_VALID_AGG = {"none", "avg", "max", "min", "sum", "count"}
_VALID_GROUP_BY = {None, "hour", "day", "month"}


def query_grain_data(
    hwdm: Optional[str] = None,
    grain_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    production_area: Optional[str] = None,
    limit: int = 20,
    agg: str = "none",
    group_by: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """查询粮仓监测数据，支持筛选、聚合与时间分组。

    参数：
        hwdm:            货位代码（模糊匹配）
        grain_name:       粮种名称（模糊匹配）
        start_date:       检测起始日期 YYYY-MM-DD
        end_date:         检测截止日期 YYYY-MM-DD
        production_area:  产地（模糊匹配）
        limit:            返回条数上限，默认 20
        agg:              聚合模式。none（默认）返回原始记录；
                          avg/max/min/sum/count 在 SQL 层聚合
        group_by:         当 agg≠none 时，按 hour/day/month 分组；
                          为 None 时返回单条汇总结果
    """
    if agg not in _VALID_AGG:
        raise ValueError(f"无效的聚合类型: {agg}，可选: {_VALID_AGG}")
    if group_by not in _VALID_GROUP_BY:
        raise ValueError(f"无效的分组维度: {group_by}，可选: {_VALID_GROUP_BY}")

    # 明细模式（默认，保持向后兼容）
    if agg == "none":
        return _query_detail(
            hwdm=hwdm, grain_name=grain_name,
            start_date=start_date, end_date=end_date,
            production_area=production_area, limit=limit,
        )

    # 聚合模式
    return _query_agg(
        agg=agg, group_by=group_by,
        hwdm=hwdm, grain_name=grain_name,
        start_date=start_date, end_date=end_date,
        production_area=production_area, limit=limit,
    )
