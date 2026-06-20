"""
Redis 会话缓存性能测试：直接对比 Redis vs MySQL 读取延迟。

用法：
    uv run python eval/eval_redis.py
"""

import time
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_redis_raw():
    """直接测 Redis GET 延迟。"""
    from src.session_cache import _get_redis
    import uuid

    cid = "perf_test_" + uuid.uuid4().hex[:8]

    # 写入测试数据到 Redis LIST
    test_msgs = [
        {"role": "user", "content": "LS/T 1211 标准是什么？"},
        {"role": "assistant", "content": "LS/T 1211 规定了粮食容重测定方法"},
    ]
    r = _get_redis()
    key = f"msgs:{cid}:all"
    r.delete(key)
    for msg in test_msgs:
        r.rpush(key, json.dumps(msg, ensure_ascii=False))
    r.setex(f"msgs:{cid}:10", 1800, json.dumps(test_msgs, ensure_ascii=False))

    # 测量 Redis GET
    r = _get_redis()
    key = f"msgs:{cid}:10"

    times = []
    for _ in range(100):
        t0 = time.time()
        data = r.get(key)
        json.loads(data)
        times.append((time.time() - t0) * 1000)

    print(f"  Redis GET:  avg={statistics.mean(times):.4f}ms  "
          f"min={min(times):.4f}ms  max={max(times):.4f}ms  (100次)")

    # 清理
    r.delete(key, f"msgs:{cid}:all")


def test_mysql_raw():
    """直接测 MySQL 读取延迟（通过 run_in_executor）。"""
    import asyncio
    import uuid

    cid = "perf_test_" + uuid.uuid4().hex[:8]

    # 先写一条以确保有数据
    from src import chat_store
    chat_store.create_conversation(title="perf", cid=cid)
    chat_store.add_message(cid, "user", "测试消息")

    # 测量 MySQL 读取（通过 run_in_executor 模拟异步路径）
    async def measure():
        loop = asyncio.get_running_loop()
        times = []
        for _ in range(20):
            t0 = time.time()
            rows = await loop.run_in_executor(None, chat_store.get_messages, cid, 10)
            times.append((time.time() - t0) * 1000)
        return times

    times = asyncio.run(measure())
    print(f"  MySQL 读:   avg={statistics.mean(times):.4f}ms  "
          f"min={min(times):.4f}ms  max={max(times):.4f}ms  (20次)")

    # 清理
    chat_store.delete_conversation(cid)


def test_cache_read_path():
    """测缓存读取完整路径（async get_messages_cached，从 Redis LIST 读）。"""
    import asyncio
    import uuid
    from src.session_cache import get_messages_cached, _get_redis

    cid = "perf_test_" + uuid.uuid4().hex[:8]
    test_msgs = [{"role": "user", "content": "测试"}] * 5
    r = _get_redis()
    key = f"msgs:{cid}:all"
    for msg in test_msgs:
        r.rpush(key, json.dumps(msg, ensure_ascii=False))
    r.expire(key, 1800)

    async def measure():
        times = []
        for _ in range(20):
            t0 = time.time()
            rows = await get_messages_cached(cid, 10)
            times.append((time.time() - t0) * 1000)
        return times

    times = asyncio.run(measure())
    print(f"  缓存读路径: avg={statistics.mean(times):.4f}ms  "
          f"min={min(times):.4f}ms  max={max(times):.4f}ms  (20次)")


def test_redis_list_write():
    """测 Redis-first 写路径（RPUSH + debounce 触发）。"""
    import asyncio
    import uuid
    from src.session_cache import add_message_redis_first

    cid = "perf_test_" + uuid.uuid4().hex[:8]

    # 确保对话存在于 MySQL（外键要求）
    from src import chat_store
    chat_store.create_conversation(title="perf", cid=cid)

    async def measure():
        times = []
        for _ in range(20):
            t0 = time.time()
            await add_message_redis_first(cid, "user", "性能测试消息内容")
            times.append((time.time() - t0) * 1000)
        return times

    times = asyncio.run(measure())
    print(f"  Redis-first 写: avg={statistics.mean(times):.4f}ms  "
          f"min={min(times):.4f}ms  max={max(times):.4f}ms  (20次)")

    # 清理
    r = _get_redis()
    r.delete(f"msgs:{cid}:all", f"msgs:{cid}:10", f"msgs:{cid}:mysql_count")
    chat_store.delete_conversation(cid)


def test_redis_list_read():
    """测 Redis LIST 读路径（LRANGE）。"""
    import asyncio
    from src.session_cache import get_messages_cached, _get_redis

    # 造数据：写入 20 条消息到 Redis
    r = _get_redis()
    cid = "perf_read_test"
    import uuid
    cid = "perf_test_" + uuid.uuid4().hex[:8]
    key = f"msgs:{cid}:all"
    for i in range(20):
        r.rpush(key, json.dumps({"role": "user", "content": f"消息{i}"}))
    r.expire(key, 1800)

    async def measure():
        times = []
        for _ in range(100):
            t0 = time.time()
            rows = await get_messages_cached(cid, 10)
            times.append((time.time() - t0) * 1000)
        return times

    times = asyncio.run(measure())
    print(f"  Redis LIST 读: avg={statistics.mean(times):.4f}ms  "
          f"min={min(times):.4f}ms  max={max(times):.4f}ms  (100次, 20条数据取10)")

    r.delete(key)


if __name__ == "__main__":
    print("Redis 会话缓存性能对比")
    print("=" * 60)

    print("\n[1] Redis 原生 GET 延迟")
    test_redis_raw()

    print("\n[2] MySQL 原生读取延迟")
    test_mysql_raw()

    print("\n[3] 旧版缓存读取路径")
    test_cache_read_path()

    print(f"\n[4] Redis-first 写路径（RPUSH + debounce）")
    test_redis_list_write()

    print(f"\n[5] Redis LIST 读路径（LRANGE）")
    test_redis_list_read()

    print(f"\n{'='*60}")
    print("  架构: Redis 热存储 + MySQL 异步批量持久化")
    print("  写路径: RPUSH (<0.1ms) → debounce 3s → MySQL batch INSERT")
    print("  读路径: LRANGE (<0.5ms) → 未命中 → MySQL → 恢复 Redis")
    print(f"{'='*60}")
