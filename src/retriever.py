"""
混合检索引擎：BM25 关键词 + 向量语义 + RRF 融合 + Cross-Encoder 精排。

流程：
  用户查询
    ├─ BM25 关键词检索 → top 20
    ├─ 向量语义检索 → top 20
    ├─ RRF 融合去重 → top 15
    └─ Cross-Encoder 精排 → top_k
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from llama_index.core.retrievers import BaseRetriever
from llama_index.core import Settings
from llama_index.core.schema import BaseNode, QueryBundle, NodeWithScore

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT / "src" / "bge-large-zh-v1.5")


def _find_local_reranker() -> str:
    """查找本地缓存的 bge-reranker-v2-m3 模型路径，不联网。"""
    base = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / "models--BAAI--bge-reranker-v2-m3" / "snapshots"
    )
    if base.exists():
        snaps = sorted(base.iterdir(), reverse=True)
        for snap in snaps:
            if (snap / "model.safetensors").exists():
                return str(snap)
    local = PROJECT_ROOT / "models" / "bge-reranker-v2-m3"
    if local.exists():
        return str(local)
    raise FileNotFoundError(
        "未找到本地 bge-reranker-v2-m3 模型。请放置于 ~/.cache/huggingface/hub/ 或项目 models/ 目录"
    )


class HybridRetriever(BaseRetriever):
    """
    混合检索器：BM25 + 向量 + RRF + Cross-Encoder。

    使用方式：
        retriever = HybridRetriever(vector_store, nodes=nodes, top_k=5)
        results = retriever.retrieve("粮食安全水分标准", top_k=5)
    """

    def __init__(self, vector_store, nodes=None, top_k=5):
        super().__init__()
        self._nodes = nodes or []
        self._bm25 = self._build_bm25(self._nodes) if self._nodes else None

        # 向量检索引擎（embed_model 由 Settings 自动注入）
        from llama_index.core import VectorStoreIndex
        self._vector_index = VectorStoreIndex.from_vector_store(vector_store)

        # Cross-Encoder（延迟加载）
        self._reranker: Optional[CrossEncoder] = None
        self._top_k = top_k

    # ============================================================
    # BM25 关键词检索
    # ============================================================

    def _tokenize(self, text: str) -> List[str]:
        """中文分词，用于 BM25。"""
        import jieba
        return list(jieba.lcut(text))

    def _build_bm25(self, nodes: List[BaseNode]) -> BM25Okapi:
        print(f"构建 BM25 索引（{len(nodes)} 条）...")
        tokenized = [self._tokenize(n.text) for n in nodes]
        print("BM25 索引就绪")
        return BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # 归一化
        scores = np.array(scores)
        if scores.max() > 0:
            scores = scores / scores.max()
        # top_k
        idxs = np.argsort(scores)[::-1][:top_k]
        return [
            {"node": self._nodes[i], "score": float(scores[i]), "source": "bm25"}
            for i in idxs if scores[i] > 0
        ]

    # ============================================================
    # 向量检索
    # ============================================================

    def _vector_search(self, query: str, top_k: int = 20) -> List[Dict]:
        retriever = self._vector_index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [
            {"node": n.node, "score": float(n.score) if n.score else 0, "source": "vector"}
            for n in nodes
        ]

    # ============================================================
    # RRF 融合
    # ============================================================

    @staticmethod
    def _rrf_fuse(results_a: List[Dict], results_b: List[Dict], k: int = 40, top_k: int = 15) -> List[Dict]:
        """
        Reciprocal Rank Fusion：合并 BM25 和向量检索结果，并标记来源。
        用内容哈希去重（而非 node_id），因为两路检索的 node_id 来源不同。
        """
        import hashlib
        def _text_id(text: str) -> str:
            return hashlib.md5(text.encode()).hexdigest()

        rrf_scores: Dict[str, float] = {}
        node_map: Dict[str, Any] = {}
        source_map: Dict[str, set] = defaultdict(set)

        for rank, item in enumerate(results_a):
            tid = _text_id(item["node"].text)
            rrf_scores[tid] = rrf_scores.get(tid, 0) + 1.0 / (k + rank + 1)
            node_map[tid] = item["node"]
            source_map[tid].add(item["source"])

        for rank, item in enumerate(results_b):
            tid = _text_id(item["node"].text)
            rrf_scores[tid] = rrf_scores.get(tid, 0) + 1.0 / (k + rank + 1)
            node_map[tid] = item["node"]
            source_map[tid].add(item["source"])

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            {
                "node": node_map[tid],
                "rrf_score": round(rrf_scores[tid], 4),
                "sources": sorted(source_map[tid]),
            }
            for tid in sorted_ids
        ]

    # ============================================================
    # Cross-Encoder 精排
    # ============================================================

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            model_path = _find_local_reranker()
            print(f"加载 Cross-Encoder（本地）: {model_path}")
            self._reranker = CrossEncoder(model_path)
        return self._reranker

    def _cross_encoder_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """交叉编码器精排：对每个 (query, doc) 对计算真实相关性分数。"""
        pairs = [[query, c["node"].text] for c in candidates]
        scores = self.reranker.predict(pairs)

        # 排序
        idxs = np.argsort(scores)[::-1][:top_k]
        return [
            {**candidates[i], "ce_score": round(float(scores[i]), 4)}
            for i in idxs
        ]

    # ============================================================
    # 统一检索入口
    # ============================================================

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """BaseRetriever 标准接口。"""
        query = query_bundle.query_str

        # 1. 双路召回
        bm25_results = self._bm25_search(query, top_k=20) if self._bm25 else []
        vec_results = self._vector_search(query, top_k=20)

        # 2. RRF 融合
        if bm25_results:
            fused = self._rrf_fuse(bm25_results, vec_results, top_k=15)
        else:
            fused = [
                {"node": r["node"], "rrf_score": r["score"], "sources": ["vector"]}
                for r in vec_results[:15]
            ]

        # 3. Cross-Encoder 精排
        ranked = self._cross_encoder_rerank(query, fused, top_k=self._top_k)

        return [
            NodeWithScore(node=r["node"], score=r.get("ce_score", 0))
            for r in ranked
        ]

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """
        便利方法：保留旧返回格式（service.py 兼容）。

        返回: [{"text": "...", "file_name": "...", "score": ..., ...}, ...]
        """
        effective_top_k = top_k if top_k is not None else self._top_k
        results = self._retrieve(QueryBundle(query_str=query))
        # 截取 top_k 条
        results = results[:effective_top_k]
        return [
            {
                "text": r.node.text,
                "file_name": r.node.metadata.get("file_name", ""),
                "file_type": r.node.metadata.get("file_type", ""),
                "page_count": r.node.metadata.get("page_count", 0),
                "score": round(r.score, 4) if r.score else 0,
            }
            for r in results
        ]
