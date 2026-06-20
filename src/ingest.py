"""
一键入库脚本：加载文档 → 分块 → 向量化 → 存入 ChromaDB。

用法：
    python src/ingest.py                  # 增量入库（跳过未变更文件）
    python src/ingest.py --full           # 全量重建（丢弃旧数据）
    python src/ingest.py --test           # 测试模式（仅 10 个文档）
    python src/ingest.py --query "粮食安全水分标准"  # 入库后测试检索
"""
import sys
import io
from pathlib import Path

# 修复 Windows GBK 编码问题：强制 stdout 使用 utf-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.service import KnowledgeBaseService
from src.loader import load_all_documents
from src.chunker import chunk_documents
from src.settings import configure_settings


def query_test(kb: KnowledgeBaseService, question: str, top_k: int = 3):
    """测试检索：混合检索，不含 LLM。"""
    results = kb.hybrid_search(question, top_k=top_k)
    print(f"\n查询: {question}")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"\n结果 {i+1} (ce_score={r.get('ce_score', 0):.4f}): {r.get('file_name', '?')}")
        print(r["text"][:300])
        print("...")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 知识库入库 (ChromaDB)")
    parser.add_argument("--query", type=str, help="入库后测试检索的问题")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回数量")
    parser.add_argument("--test", action="store_true", help="测试模式：仅用前 10 个文档")
    parser.add_argument("--full", action="store_true", help="全量重建（默认增量模式）")
    args = parser.parse_args()

    configure_settings()

    kb = KnowledgeBaseService()

    if args.test:
        print("[测试模式] 仅加载前 10 个文档\n")
        all_docs = load_all_documents()
        test_docs = all_docs[:10]

        kb._get_vector_store(overwrite=True)
        kb.embed_model
        total = 0
        for i, doc in enumerate(test_docs):
            nodes = chunk_documents([doc])
            kb._add_nodes_to_collection(nodes)
            total += len(nodes)
            fname = doc.metadata.get("file_name", "?")
            print(f"  [{i+1}/10] {fname} ({len(nodes)} chunks)")

        print(f"\n测试入库完成! 共 {total} chunks")
    else:
        total = kb.ingest_all(full_rebuild=args.full)
        if total == 0 and not args.full:
            print("\n所有文件均未变更，无需入库。")
        elif args.full:
            print(f"\n全量重建完成! 共 {total} chunks")
        else:
            print(f"\n增量入库完成! 共 {total} chunks")

    if args.query:
        query_test(kb, args.query, args.top_k)
