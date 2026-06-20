"""
语义缓存管理：对相似问题返回缓存答案，减少重复 LLM 调用。

工作原理：
  新问题 → BGE embedding → FAISS 向量相似度搜索
    │
    ├─ 相似度 > 阈值 → 命中缓存，直接返回答案
    └─ 相似度 ≤ 阈值 → 正常走 Agent，完成后存入缓存
"""
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT / "src" / "bge-large-zh-v1.5")


class SemanticCache:
    """基于 FAISS 的语义缓存。"""

    def __init__(self, similarity_threshold: float = 0.92, max_size: int = 500):
        self.threshold = similarity_threshold
        self.max_size = max_size

        # 延迟加载 embedding 模型（首次使用才加载，约 1.3GB 内存）
        self._embed_model = None
        self._index: Optional[faiss.IndexFlatIP] = None  # 内积索引（余弦相似度）
        self._questions: List[str] = []   # 原始问题文本
        self._answers: List[str] = []     # 缓存的答案

    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(MODEL_DIR)
            # 确保输出归一化，使内积等价于余弦相似度
            self._embed_model.encode("init", normalize_embeddings=True)
        return self._embed_model

    def _ensure_index(self, dim: int):
        """初始化 FAISS 索引（首次使用时创建）。"""
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine (归一化后)

    def _encode(self, text: str) -> np.ndarray:
        """将文本编码为归一化向量。"""
        vec = self.embed_model.encode(
            text,
            normalize_embeddings=True,  # L2 归一化 → 内积 = 余弦相似度
            show_progress_bar=False,
        )
        return vec.astype(np.float32).reshape(1, -1)

    def search(self, query: str) -> Optional[str]:
        """
        检索缓存。返回相似度超过阈值的缓存答案，无匹配则返回 None。
        """
        if self._index is None or self._index.ntotal == 0:
            return None

        query_vec = self._encode(query)
        scores, indices = self._index.search(query_vec, 1)  # 找最相似的 1 条

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        if best_score >= self.threshold and best_idx >= 0:
            return self._answers[best_idx]

        return None

    def add(self, query: str, answer: str):
        """
        将查询-答案对存入缓存。超过最大容量时淘汰最旧的。
        """
        # 不缓存太短的答案（闲聊、问候）
        if len(answer.strip()) < 20:
            return

        # 去重：如果已有高度相似的查询，跳过
        if self.search(query) is not None:
            return

        vec = self._encode(query)
        self._ensure_index(vec.shape[1])

        # 超过容量限制：删除最旧的记录
        if self._index.ntotal >= self.max_size:
            self._clear_oldest()

        self._index.add(vec)
        self._questions.append(query)
        self._answers.append(answer)

    def _clear_oldest(self):
        """淘汰最旧的一条缓存。"""
        if self._index.ntotal == 0:
            return
        self._questions.pop(0)
        self._answers.pop(0)
        self._rebuild_index()

    def remove(self, query: str) -> int:
        """移除与 query 相似度超过阈值的缓存条目。返回移除条数。"""
        if self._index is None or self._index.ntotal == 0:
            return 0

        query_vec = self._encode(query)
        # 查找所有相似度超过阈值的条目
        scores, indices = self._index.search(query_vec, self._index.ntotal)

        remove_set = set()
        for score, idx in zip(scores[0], indices[0]):
            if float(score) >= self.threshold and int(idx) >= 0:
                remove_set.add(int(idx))

        if not remove_set:
            return 0

        # 从列表尾部向头部删除，避免索引偏移
        for i in sorted(remove_set, reverse=True):
            self._questions.pop(i)
            self._answers.pop(i)

        # 重建索引
        self._rebuild_index()
        return len(remove_set)

    def _rebuild_index(self):
        """重建 FAISS 索引（删除操作后的通用重建逻辑）。"""
        dim = self._index.d if self._index is not None else 0
        if dim == 0:
            return
        self._index = faiss.IndexFlatIP(dim)
        if self._questions:
            vectors = np.vstack([self._encode(q) for q in self._questions])
            self._index.add(vectors)

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        return {
            "cached_count": len(self._questions),
            "max_size": self.max_size,
            "threshold": self.threshold,
            "hit_rate": self._hit_rate if hasattr(self, '_hit_rate') else 0.0,
        }


# 全局单例（Agent 启动时初始化一次即可）
_cache: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
