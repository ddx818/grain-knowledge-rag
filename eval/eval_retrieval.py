"""
检索评估脚本：Baseline vs 混合检索对比。

  Baseline (纯向量):   kb.search() → ChromaDB 向量相似度，无 BM25/无 RRF/无 Cross-Encoder
  混合检索 (Our):      kb.hybrid_search() → BM25(20) + 向量(20) → RRF(15) → Cross-Encoder(top_k)

用法：
    uv run python eval/eval_retrieval.py            # 默认 top_k=5
    uv run python eval/eval_retrieval.py --top_k 3  # 指定 top_k
"""

import json
import sys
import time
import math
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.service import KnowledgeBaseService


def load_questions() -> list[dict]:
    path = Path(__file__).parent / "test_questions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def keyword_hit_score(keywords: list[str], text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords) if keywords else 0.0


def binary_relevance(keywords: list[str], text: str) -> int:
    text_lower = text.lower()
    return 1 if any(kw.lower() in text_lower for kw in keywords) else 0


def hit_at_k(results: list[dict], keywords: list[str], k: int) -> int:
    for r in results[:k]:
        if binary_relevance(keywords, r["text"]):
            return 1
    return 0


def mrr(results: list[dict], keywords: list[str]) -> float:
    for rank, r in enumerate(results, 1):
        if binary_relevance(keywords, r["text"]):
            return 1.0 / rank
    return 0.0


def ndcg(results: list[dict], keywords: list[str], k: int) -> float:
    relevance = [keyword_hit_score(keywords, r["text"]) for r in results[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(top_k: int = 5):
    questions = load_questions()
    kb = KnowledgeBaseService()

    vec_stats = {"mrr": 0.0, "ndcg": 0.0, "h1": 0, "h3": 0, "h5": 0}
    hyb_stats = {"mrr": 0.0, "ndcg": 0.0, "h1": 0, "h3": 0, "h5": 0}
    details = []

    print(f"\n{'='*70}")
    print(f"  检索评估 · {len(questions)} 题 · top_k={top_k}")
    print(f"  对比: 纯向量检索 vs 混合检索 (BM25+向量+RRF+Reranker)")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:50]}...")
        keywords = q.get("expected_keywords", [])
        per_q = {}

        for label, results_fn, stats in [
            ("vector", lambda: kb.search(q["question"], top_k=top_k), vec_stats),
            ("hybrid", lambda: kb.hybrid_search(q["question"], top_k=top_k), hyb_stats),
        ]:
            t0 = time.time()
            try:
                results = results_fn()
            except Exception as e:
                print(f"  {label}: ERR - {e}")
                continue
            elapsed = (time.time() - t0) * 1000

            m = mrr(results, keywords)
            n = ndcg(results, keywords, top_k)
            stats["mrr"] += m
            stats["ndcg"] += n
            for k in [1, 3, 5]:
                if k <= top_k and hit_at_k(results, keywords, k):
                    stats[f"h{k}"] += 1

            per_q[label] = {"mrr": m, "ndcg": n, "results": results,
                            "elapsed": elapsed, "h1": hit_at_k(results, keywords, 1)}
            print(f"  {label:>7}: MRR={m:.3f}  NDCG={n:.3f}  H@1={per_q[label]['h1']}  ({elapsed:.0f}ms)")

        # 保存明细
        hyb_res = per_q.get("hybrid", {}).get("results", [])
        top_items = [{
            "text": r["text"][:300],
            "file_name": r.get("file_name", ""),
            "ce_score": r.get("ce_score"),
            "sources": r.get("sources", []),
        } for r in hyb_res]

        details.append({
            "id": q["id"],
            "question": q["question"],
            "category": q.get("category", ""),
            "vector_mrr": round(per_q.get("vector", {}).get("mrr", 0), 4),
            "hybrid_mrr": round(per_q.get("hybrid", {}).get("mrr", 0), 4),
            "hybrid_ndcg": round(per_q.get("hybrid", {}).get("ndcg", 0), 4),
            "ce_score_top1": hyb_res[0].get("ce_score") if hyb_res else None,
            "top_results": top_items,
        })

    # ── 汇总 ──
    N = len(details)
    v = {k: vec_stats[k] / N for k in ["mrr", "ndcg"]}
    h = {k: hyb_stats[k] / N for k in ["mrr", "ndcg"]}
    mrr_boost = (h["mrr"] - v["mrr"]) / v["mrr"] * 100 if v["mrr"] > 0 else 0
    ndcg_boost = (h["ndcg"] - v["ndcg"]) / v["ndcg"] * 100 if v["ndcg"] > 0 else 0

    print(f"\n{'='*70}")
    print(f"  评估汇总 ({N} 题, top_k={top_k})")
    print(f"{'='*70}")
    print(f"  {'指标':<12} {'向量检索':>12} {'混合检索':>12} {'提升':>12}")
    print(f"  {'-'*48}")
    print(f"  {'MRR':<12} {v['mrr']:>12.4f} {h['mrr']:>12.4f} {mrr_boost:>+11.1f}%")
    print(f"  {'NDCG@'+str(top_k):<12} {v['ndcg']:>12.4f} {h['ndcg']:>12.4f} {ndcg_boost:>+11.1f}%")
    print(f"  {'Hit@1':<12} {vec_stats['h1']:>11}/{N}  {hyb_stats['h1']:>11}/{N}  {hyb_stats['h1']-vec_stats['h1']:>+11}")
    print(f"  {'Hit@3':<12} {vec_stats['h3']:>11}/{N}  {hyb_stats['h3']:>11}/{N}")
    print(f"  {'Hit@5':<12} {vec_stats['h5']:>11}/{N}  {hyb_stats['h5']:>11}/{N}")
    print(f"{'='*70}\n")

    # ── 保存 ──
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_{ts}.json"
    out_path.write_text(json.dumps({
        "config": {"top_k": top_k, "questions": N},
        "summary": {
            "vector": {"mrr": round(v["mrr"], 4), f"ndcg@{top_k}": round(v["ndcg"], 4), "hit@1": f"{vec_stats['h1']}/{N}"},
            "hybrid": {"mrr": round(h["mrr"], 4), f"ndcg@{top_k}": round(h["ndcg"], 4), "hit@1": f"{hyb_stats['h1']}/{N}",
                       "mrr_boost": f"{mrr_boost:+.1f}%", "ndcg_boost": f"{ndcg_boost:+.1f}%"},
        },
        "details": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细已保存: {out_path}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    run_evaluation(top_k=args.top_k)
