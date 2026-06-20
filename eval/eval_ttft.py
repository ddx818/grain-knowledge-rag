"""
TTFT 测评：对比流式(SSE)与非流式的首字延迟。
"""

import time
import requests
import statistics
import uuid

API = "http://localhost:8000"

# Q1: 闲聊（不调工具）  Q2: 检索（调 search_kb）
QUESTIONS = [
    ("检索", "LS/T 1211 标准规定了什么检测方法？"),
    ("检索", "粮食安全水分标准是什么？"),
    ("检索", "GB/T 29890 标准中粮温检测频率的规定是什么？"),
    ("检索", "脂肪酸值测定中KOH标准溶液的浓度是多少？"),
    ("检索", "粮仓气密性检测500Pa降至250Pa的时间要求？"),
]


def test_streaming(question: str, label: str) -> dict:
    sid = str(uuid.uuid4())
    print(f"  连接 {label}...")
    t0 = time.time()
    try:
        resp = requests.post(f"{API}/api/chat/stream", json={
            "message": question, "session_id": sid
        }, stream=True, timeout=180)

        first_event_time = None
        last_event_time = None
        answer_chars = 0
        event_count = 0

        buffer = ""
        for chunk in resp.iter_content(chunk_size=None):
            if first_event_time is None:
                first_event_time = (time.time() - t0) * 1000
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                event_part, buffer = buffer.split("\n\n", 1)
                event_count += 1
                for line in event_part.split("\n"):
                    if line.startswith("data: ") and "event: answer" in event_part:
                        answer_chars += len(line[6:])
                last_event_time = (time.time() - t0) * 1000
        resp.close()

        return {"ttft_ms": first_event_time or 0, "total_ms": last_event_time or 0,
                "events": event_count, "answer_chars": answer_chars}
    except Exception as e:
        print(f"  {label} 失败: {e}")
        return None


def test_non_streaming(question: str, label: str) -> dict:
    sid = str(uuid.uuid4())
    print(f"  连接 {label}...")
    t0 = time.time()
    try:
        resp = requests.post(f"{API}/api/chat", json={
            "message": question, "session_id": sid
        }, timeout=180)
        total_ms = (time.time() - t0) * 1000
        data = resp.json()
        return {"total_ms": total_ms, "answer_len": len(data.get("answer", ""))}
    except Exception as e:
        print(f"  {label} 失败: {e}")
        return None


def run():
    print("TTFT 测评：流式(SSE) vs 非流式")
    print("=" * 60)

    stream_ttfts = []
    non_stream_times = []

    for qtype, question in QUESTIONS:
        print(f"\n[{qtype}] {question[:40]}...")

        ns = test_non_streaming(question, qtype)
        if ns:
            non_stream_times.append(ns["total_ms"])
            print(f"  非流式: {ns['total_ms']:.0f}ms (回答{ns['answer_len']}字)")

        st = test_streaming(question, qtype)
        if st:
            stream_ttfts.append(st["ttft_ms"])
            print(f"  流 式:  TTFT={st['ttft_ms']:.0f}ms  ({st['answer_chars']}字, {st['events']}事件, {st['total_ms']:.0f}ms完成)")

    # 汇总
    if stream_ttfts and non_stream_times:
        avg_ns = statistics.mean(non_stream_times)
        avg_ttft = statistics.mean(stream_ttfts)
        speedup = avg_ns / avg_ttft if avg_ttft > 0 else 0
        print(f"\n{'='*60}")
        print(f"  汇总 ({len(stream_ttfts)} 题)")
        print(f"{'='*60}")
        print(f"  非流式 平均完整响应时间:  {avg_ns:.0f} ms")
        print(f"  流 式 平均首字到达 (TTFT): {avg_ttft:.0f} ms")
        print(f"  首字提前倍数:              {speedup:.1f}x")
        print(f"  {'='*60}")


if __name__ == "__main__":
    run()
