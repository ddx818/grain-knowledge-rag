"""
Ragas 评估入口脚本 —— 语义级 RAG 质量评估。

流程：加载 100 题 → 逐题调 QA 服务 → 收集数据 → Ragas 评判 → 终端输出 + JSON 落盘。

用法：
    uv run python eval/eval_ragas.py                 # 全量 100 题
    uv run python eval/eval_ragas.py --limit 10      # 前 10 题快速验证
    uv run python eval/eval_ragas.py --metrics retrieval   # 仅检索指标
    uv run python eval/eval_ragas.py --metrics generation  # 仅生成指标
"""
import json
import os
import sys
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.service import KnowledgeBaseService
from src.qa import QAService
from eval.ragas_evaluator import RagasEvaluator


def load_questions(limit: int | None = None) -> list[dict]:
    """加载测试题目。"""
    path = Path(__file__).parent / "eval_ragas_questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    return questions


def collect_data(questions: list[dict], kb: KnowledgeBaseService, qa: QAService) -> list[dict]:
    """逐题执行 QA，收集 user_input / response / retrieved_contexts / reference。"""
    dataset = []
    for q in tqdm(questions, desc="收集 QA 数据", unit="题"):
        try:
            result = qa.ask(q["question"])
            contexts = [c["text"] for c in result.get("contexts", [])]
            response = result.get("answer", "")
        except Exception as e:
            print(f"  [ERROR] Q{q['id']}: {e}")
            contexts = []
            response = ""

        dataset.append({
            "user_input": q["question"],
            "response": response,
            "retrieved_contexts": contexts,
            "reference": q["reference"],
            "id": q["id"],
            "category": q.get("category", ""),
        })
    return dataset


def compute_stats(scores: list[float]) -> dict:
    """计算均值/中位数/P25/P75/标准差。"""
    if not scores:
        return {"mean": None, "median": None, "p25": None, "p75": None, "std": None, "count": 0}
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    return {
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "p25": round(sorted_scores[n // 4], 4),
        "p75": round(sorted_scores[3 * n // 4], 4),
        "std": round(statistics.stdev(scores), 4) if n >= 2 else 0,
        "count": n,
    }


def compute_category_stats(dataset: list[dict], all_scores: dict[str, list[float]]) -> dict:
    """按 category 分组计算各指标均值。"""
    cat_indices = defaultdict(list)
    for i, item in enumerate(dataset):
        cat_indices[item.get("category", "未分类")].append(i)

    result = {}
    for cat, indices in cat_indices.items():
        result[cat] = {"count": len(indices)}
        for metric, scores in all_scores.items():
            cat_scores = [scores[i] for i in indices if i < len(scores) and scores[i] is not None]
            if cat_scores:
                result[cat][metric] = round(statistics.mean(cat_scores), 4)
    return result


def print_results(metrics_stats: dict, category_stats: dict, total: int, model: str):
    """终端输出汇总表和分类统计。"""
    METRICS_ORDER = [
        "context_precision", "context_recall",
        "faithfulness", "answer_relevancy", "factual_correctness",
    ]
    METRIC_LABELS = {
        "context_precision": "ContextPrecision",
        "context_recall": "ContextRecall",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "AnswerRelevancy",
        "factual_correctness": "FactualCorrectness",
    }

    print()
    print("═" * 72)
    print(f"  Ragas 评估结果 · {total} 题 · {model}")
    print("═" * 72)
    print(f"  {'指标':<24} {'均值':>8} {'中位数':>8} {'P25':>8} {'P75':>8} {'Std':>8}")
    print("─" * 72)

    composite = 0
    n_metrics = 0
    for key in METRICS_ORDER:
        if key not in metrics_stats:
            continue
        s = metrics_stats[key]
        if s["mean"] is None:
            continue
        label = METRIC_LABELS.get(key, key)
        print(f"  {label:<24} {s['mean']:>8.3f} {s['median']:>8.3f} "
              f"{s['p25']:>8.3f} {s['p75']:>8.3f} {s['std']:>8.3f}")
        composite += s["mean"]
        n_metrics += 1

    print("─" * 72)
    if n_metrics > 0:
        print(f"  综合得分: {composite / n_metrics:.3f}")
    print("═" * 72)
    print()

    if category_stats:
        print("分类汇总:")
        print(f"  {'类别':<12} {'数量':>6}", end="")
        for key in METRICS_ORDER:
            if key in metrics_stats:
                print(f"  {METRIC_LABELS.get(key, key):>18}", end="")
        print()
        print("─" * (24 + 20 * n_metrics))
        for cat, info in sorted(category_stats.items()):
            print(f"  {cat:<12} {info['count']:>6}", end="")
            for key in METRICS_ORDER:
                if key in info:
                    print(f"  {info[key]:>18.3f}", end="")
            print()
        print()


def save_results(metrics_stats: dict, category_stats: dict, dataset: list[dict],
                 total: int, model: str) -> Path:
    """保存 JSON 结果文件。"""
    out_dir = Path(__file__).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_ragas_{ts}.json"

    details = []
    for item in dataset:
        details.append({
            "id": item["id"],
            "question": item["user_input"],
            "category": item.get("category", ""),
            "reference": item["reference"],
            "response": item["response"][:300],
        })

    out_path.write_text(json.dumps({
        "config": {"total": total, "model": model},
        "summary": metrics_stats,
        "category_summary": category_stats,
        "details": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ragas RAG 质量评估")
    parser.add_argument("--limit", type=int, default=None, help="限制题目数量（默认全部）")
    parser.add_argument("--metrics", choices=["all", "retrieval", "generation"],
                        default="all", help="评估指标组（默认 all）")
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)

    print(f"\n加载 {len(questions)} 道测试题")
    kb = KnowledgeBaseService()
    qa = QAService(kb)

    # Step 1: 收集数据
    dataset = collect_data(questions, kb, qa)

    # Step 2: Ragas 评估
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    evaluator = RagasEvaluator()

    if args.metrics in ("all", "retrieval"):
        print(f"\n{'='*60}")
        print("  阶段 1/2: 检索质量评估 (ContextPrecision + ContextRecall)")
        print(f"{'='*60}")
        retrieval_scores = evaluator.evaluate_retrieval(dataset)
    else:
        retrieval_scores = {}

    if args.metrics in ("all", "generation"):
        print(f"\n{'='*60}")
        print("  阶段 2/2: 生成质量评估 (Faithfulness + AnswerRelevancy + FactualCorrectness)")
        print(f"{'='*60}")
        generation_scores = evaluator.evaluate_generation(dataset)
    else:
        generation_scores = {}

    all_scores = {**retrieval_scores, **generation_scores}

    # Step 3: 统计与输出
    metrics_stats = {k: compute_stats(v) for k, v in all_scores.items()}
    category_stats = compute_category_stats(dataset, all_scores)
    print_results(metrics_stats, category_stats, len(dataset), model)

    # Step 4: 保存
    out_path = save_results(metrics_stats, category_stats, dataset,
                            len(dataset), model)
    print(f"明细已保存: {out_path}\n")


if __name__ == "__main__":
    main()
