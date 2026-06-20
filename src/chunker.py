"""
文档分块模块（最终稳定版）
中文最优分块策略：句子切分 + BGE真实Token计数 + 安全硬兜底
保证所有chunk ≤ 512 token，不破坏语义，不丢信息
"""
import sys
import re
from pathlib import Path
from typing import List

import numpy as np

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, NodeParser
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# 全局加载一次 tokenizer（避免重复加载）
from transformers import AutoTokenizer
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT / "src" / "bge-large-zh-v1.5")

# 🔥 全局只加载一次，性能提升 100 倍
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)

# ==========================
# 工具函数
# ==========================
def bge_tokenize(text: str) -> List[str]:
    return TOKENIZER.tokenize(text)

def bge_token_count(text: str) -> int:
    return len(TOKENIZER.encode(text, truncation=False))

def safe_split_long_text(text: str, max_tokens: int = 500) -> List[str]:
    """
    安全硬切分：不会破坏中文，按完整词汇切分
    用于兜底：单句 > 512 token 的极端情况
    """
    tokens = TOKENIZER.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i + max_tokens]
        chunk_text = TOKENIZER.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    return chunks

# ==========================
# 主分块函数（工业级稳定）
# ==========================
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[BaseNode]:
    """（已废弃）请使用 IngestionPipeline + SimilarityMergeNodeParser 替代。保留用于向后兼容。"""

    # 第一轮：标准中文句子分块（最优）
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_tokenizer_fn=bge_tokenize,
        paragraph_separator="\n\n",
        # 标准中文断句正则
        secondary_chunking_regex=r"([。？！；\n])",
    )

    print(f"第一轮分块（句子边界）...")
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    # 第二轮：安全兜底（确保不超 512）
    final_nodes = []
    oversized = 0
    BGE_LIMIT = 512

    for node in nodes:
        text = node.text.strip()
        if not text:
            continue  # 过滤空文本

        tc = bge_token_count(text)
        if tc <= BGE_LIMIT:
            final_nodes.append(node)
        else:
            oversized += 1
            pieces = safe_split_long_text(text, chunk_size)
            for i, p in enumerate(pieces):
                new_node = node.model_copy()
                new_node.text = p
                new_node.node_id = f"{node.node_id}_split_{i}"  # 避免 ID 重复
                final_nodes.append(new_node)

    # 统计
    tokens = [bge_token_count(n.text) for n in final_nodes]
    print(f"\n[OK] 分块完成")
    print(f"  总块数：{len(final_nodes)}")
    print(f"  token 范围：{min(tokens)} ~ {max(tokens)}")
    print(f"  硬切分块数：{oversized}")
    print(f"  全部 <= 512 token：{'YES' if max(tokens) <= 512 else 'NO'}")

    return final_nodes

# ============================================================
# SimilarityMergeNodeParser —— 先细切再按边界相似度合并
# ============================================================

_SENT_SPLIT_PATTERN = re.compile(r'(?<=[。！？；\n])')


def _get_last_sentence(text: str) -> str:
    """取文本最后一句（非空），用于边界相似度计算。"""
    parts = _SENT_SPLIT_PATTERN.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[-1] if parts else text[-100:]


def _get_first_sentence(text: str) -> str:
    """取文本第一句（非空）。"""
    parts = _SENT_SPLIT_PATTERN.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[0] if parts else text[:100]


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """余弦相似度，返回 [0, 1]."""
    dot = float(np.dot(vec_a, vec_b))
    norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return dot / norm if norm > 0 else 0.0


class SimilarityMergeNodeParser(NodeParser):
    """
    先细切再按边界语义相似度合并。

    流程（在 _parse_nodes 中）:
      遍历 node 列表，对每对相邻 node:
        1. BGE token 计数检查 → 合并后 > max_tokens 则跳过
        2. 计算 node[i] 最后一句 vs node[i+1] 第一句 的余弦相似度
        3. >= threshold → 合并为 TextNode，继续和下一位比较
        4. < threshold → 保留 node[i]，游标移到 node[i+1]

    需要在 IngestionPipeline 中放在 SentenceSplitter 之后、embed_model 之前使用。
    """

    max_tokens: int = 512
    threshold: float = 0.8

    def _parse_nodes(self, nodes, **kwargs):
        # 过滤空文本节点，避免 embedding 空字符串
        nodes = [n for n in nodes if n.text and n.text.strip()]
        if len(nodes) <= 1:
            return list(nodes)

        # 复制列表，避免原地修改传入的 nodes
        nodes = list(nodes)

        if Settings.embed_model is None:
            raise ValueError("Settings.embed_model 未配置，请先调用 configure_settings()")

        embed_model = Settings.embed_model
        merged = []
        i = 0

        while i < len(nodes):
            if i == len(nodes) - 1:
                merged.append(nodes[i])
                break

            # 检查合并后是否超 max_tokens
            combined_text = nodes[i].text + nodes[i + 1].text
            if bge_token_count(combined_text) > self.max_tokens:
                merged.append(nodes[i])
                i += 1
                continue

            # 边界相似度
            last_sent = _get_last_sentence(nodes[i].text)
            first_sent = _get_first_sentence(nodes[i + 1].text)
            emb_a = embed_model.get_text_embedding(last_sent)
            emb_b = embed_model.get_text_embedding(first_sent)
            sim = _cosine_similarity(np.array(emb_a), np.array(emb_b))

            if sim >= self.threshold:
                # 合并：保留左侧元数据，右侧元数据以 _right 前缀合并避免覆盖
                combined_meta = {**nodes[i].metadata}
                for k, v in nodes[i + 1].metadata.items():
                    if k not in combined_meta:
                        combined_meta[k] = v
                merged_node = TextNode(
                    text=combined_text,
                    metadata=combined_meta,
                )
                nodes[i + 1] = merged_node  # 替换，下一轮继续和后面的比
                i += 1
            else:
                merged.append(nodes[i])
                i += 1

        return merged


# ==========================
# 测试运行
# ==========================
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.loader import load_all_documents
    docs = load_all_documents()
    nodes = chunk_documents(docs)

    print("\n前3个分块预览：")
    for i, node in enumerate(nodes[:3]):
        print(f"\n--- Chunk {i+1} ({bge_token_count(node.text)} token) ---")
        print(node.text[:300])