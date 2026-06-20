"""
会话级 Redis 消息存储。

架构：Redis 为热存储（读写主路径），MySQL 为冷持久化（异步批量刷入）。

  - 写：RPUSH 到 Redis LIST → debounce 3 秒 → 批量 INSERT MySQL
  - 读：Redis LRANGE → 命中返回；未命中 → MySQL 恢复 → 返回
  - 降级：Redis 不可用时，读写均直接走 MySQL（等价于旧版行为）
"""

import json
import os
import logging
import asyncio
from datetime import datetime
from typing import Optional

logger = logging.getLogger("session_cache")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "1800"))
FLUSH_DEBOUNCE = float(os.getenv("REDIS_FLUSH_DEBOUNCE", "3.0"))

_redis: Optional["Redis"] = None  # type: ignore
_checked = False

# 每个会话的 debounce 排空定时器
_flush_timers: dict[str, asyncio.Task] = {}


def _get_redis():
    global _redis, _checked
    if _redis is not None:
        return _redis
    if _checked:
        return None

    try:
        from redis import Redis
        client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                       socket_connect_timeout=0.5, socket_timeout=0.5,
                       health_check_interval=30, decode_responses=True)
        client.ping()
        _redis = client
        logger.info(f"Redis 已连接 {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.warning(f"Redis 不可用 ({e})，降级为 MySQL 直读直写")
    finally:
        _checked = True

    return _redis


# ═══════════════════════════════════════════════════════════════
# 写路径：Redis 热存储 + debounce 异步刷 MySQL
# ═══════════════════════════════════════════════════════════════

async def add_message_redis_first(cid: str, role: str, content: str):
    """
    写消息：优先 RPUSH 到 Redis LIST，触发 debounce 异步批量刷 MySQL。

    Redis 不可用时降级为直接写 MySQL（等价于旧版 add_message）。
    """
    r = _get_redis()
    if r is None:
        # 降级路径：直接走 MySQL
        from src import chat_store
        await chat_store.add_message(cid, role, content)
        return

    msg = json.dumps(
        {"role": role, "content": content, "ts": datetime.now().isoformat()},
        ensure_ascii=False,
    )
    key = f"msgs:{cid}:all"

    try:
        r.rpush(key, msg)
        r.expire(key, CACHE_TTL)
    except Exception:
        pass

    # 触发 debounce 异步刷 MySQL
    _schedule_flush(cid)


def _schedule_flush(cid: str):
    """取消旧的 debounce 定时器，创建新的。"""
    if cid in _flush_timers:
        _flush_timers[cid].cancel()

    async def _delayed_flush():
        try:
            await asyncio.sleep(FLUSH_DEBOUNCE)
            await _flush_to_mysql(cid)
        except asyncio.CancelledError:
            pass
        finally:
            _flush_timers.pop(cid, None)

    _flush_timers[cid] = asyncio.create_task(_delayed_flush())


async def _flush_to_mysql(cid: str):
    """将 Redis 中尚未持久化的消息批量写入 MySQL。"""
    from src import chat_store

    r = _get_redis()
    if r is None:
        return

    msg_key = f"msgs:{cid}:all"
    count_key = f"msgs:{cid}:mysql_count"

    try:
        all_raw = r.lrange(msg_key, 0, -1)
        if not all_raw:
            return
        all_messages = [json.loads(m) for m in all_raw]

        persisted = int(r.get(count_key) or 0)
        new_messages = all_messages[persisted:]
        if not new_messages:
            return

        batch = [
            {"role": m["role"], "content": m["content"]}
            for m in new_messages
        ]
        await chat_store.add_messages_batch(cid, batch)

        r.set(count_key, len(all_messages))
        logger.debug(f"flush {cid}: {len(batch)} 条消息写入 MySQL")
    except Exception as e:
        logger.warning(f"flush {cid} 失败: {e}")


# ═══════════════════════════════════════════════════════════════
# 关闭时强制刷数据
# ═══════════════════════════════════════════════════════════════

async def flush_all_pending():
    """应用关闭时取消所有 debounce 定时器，立即刷数据到 MySQL。"""
    timers = list(_flush_timers.items())
    for cid, task in timers:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # 遍历所有会话的 Redis LIST，刷未持久化的数据
    r = _get_redis()
    if r is None:
        return

    cids = set()
    try:
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match="msgs:*:all", count=100)
            for key in keys:
                cid = key.removeprefix("msgs:").removesuffix(":all")
                cids.add(cid)
            if cursor == 0:
                break
    except Exception:
        return

    for cid in cids:
        try:
            await _flush_to_mysql(cid)
        except Exception:
            pass

    logger.info(f"关闭前已刷 {len(cids)} 个会话的消息到 MySQL")


# ═══════════════════════════════════════════════════════════════
# 读路径：Redis LIST 优先，MySQL 兜底 + 恢复
# ═══════════════════════════════════════════════════════════════

async def get_messages_cached(cid: str, limit: int = 10) -> list[dict]:
    """获取消息历史。优先读 Redis LIST，未命中则从 MySQL 恢复。"""
    r = _get_redis()
    if r is None:
        from src import chat_store
        return await chat_store.get_messages(cid, limit)

    msg_key = f"msgs:{cid}:all"

    # ① 尝试 Redis
    try:
        raw = r.lrange(msg_key, -limit, -1)
        if raw:
            return [
                {"role": m["role"], "content": m["content"]}
                for item in raw
                if (m := json.loads(item))
            ]
    except Exception:
        pass

    # ② Redis 无数据 → MySQL 恢复
    from src import chat_store
    rows = await chat_store.get_messages(cid, limit)

    if rows and r:
        try:
            # 恢复 Redis LIST
            pipe = r.pipeline()
            pipe.delete(msg_key)
            for row in rows:
                pipe.rpush(msg_key, json.dumps(row, ensure_ascii=False))
            pipe.expire(msg_key, CACHE_TTL)
            pipe.set(f"msgs:{cid}:mysql_count", len(rows))
            pipe.execute()
        except Exception:
            pass

    return rows


# ═══════════════════════════════════════════════════════════════
# 会话删除（清除 Redis + MySQL）
# ═══════════════════════════════════════════════════════════════

def invalidate_messages(cid: str):
    """删除某会话的 Redis 消息数据 + 取消待执行的 flush 定时器。"""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(f"msgs:{cid}:all", f"msgs:{cid}:mysql_count")
    except Exception:
        pass
    if cid in _flush_timers:
        _flush_timers[cid].cancel()
        _flush_timers.pop(cid, None)


def invalidate_conversation(cid: str):
    """删除单个对话元信息缓存。"""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(f"conv:{cid}")
    except Exception:
        pass


def invalidate_conversation_list():
    """删除对话列表缓存。"""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete("conv_list")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# 对话元信息缓存（不变）
# ═══════════════════════════════════════════════════════════════

async def get_conversation_cached(cid: str) -> Optional[dict]:
    from src import chat_store

    r = _get_redis()
    if r is None:
        return await chat_store.get_conversation(cid)

    key = f"conv:{cid}"
    try:
        data = r.get(key)
        if data is not None:
            return json.loads(data)
    except Exception:
        pass

    conv = await chat_store.get_conversation(cid)
    if conv is not None:
        try:
            r.setex(key, CACHE_TTL, json.dumps(conv, ensure_ascii=False))
        except Exception:
            pass
    return conv


async def get_conversation_list_cached() -> list[dict]:
    from src import chat_store

    r = _get_redis()
    if r is None:
        return await chat_store.list_conversations()

    key = "conv_list"
    try:
        data = r.get(key)
        if data is not None:
            return json.loads(data)
    except Exception:
        pass

    rows = await chat_store.list_conversations()
    try:
        r.setex(key, CACHE_TTL, json.dumps(rows, ensure_ascii=False))
    except Exception:
        pass
    return rows


# ═══════════════════════════════════════════════════════════════
# 异步便捷封装（委托给 chat_store，供非消息场景使用）
# ═══════════════════════════════════════════════════════════════

async def add_message_async(cid: str, role: str, content: str):
    """兼容旧接口：内部走 Redis-first 写路径。"""
    await add_message_redis_first(cid, role, content)


async def create_conversation_async(title: str = "新对话", cid: str = None) -> dict:
    from src import chat_store
    return await chat_store.create_conversation(title, cid)


async def get_conversation_async(cid: str) -> Optional[dict]:
    from src import chat_store
    return await chat_store.get_conversation(cid)


async def update_title_async(cid: str, title: str):
    from src import chat_store
    await chat_store.update_title(cid, title)


async def delete_conversation_async(cid: str):
    from src import chat_store
    await chat_store.delete_conversation(cid)
