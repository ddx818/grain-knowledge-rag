"""
Markdown 语义分块器 —— 配合 MinerU 结构化输出使用。

三阶段流水线：
  1. 原子解析（Markdown → heading/text/table/formula/list/image 原子）
  2. 文本原子递归切分（复用 SentenceSplitter，受保护原子跳过）
  3. 相似度合并（复用 SimilarityMergeNodeParser 逻辑）

受保护原子（table/formula_block/formula_inline/image）不会被切分，
保证表格 HTML 结构和公式 LaTeX 的完整性。

对纯文本文档（无 Markdown 结构）自动降级，行为等价于
SentenceSplitter + SimilarityMergeNodeParser。
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter, NodeParser
from llama_index.core.schema import BaseNode, TextNode

from src.chunker import (
    TOKENIZER,
    bge_token_count,
    _get_last_sentence,
    _get_first_sentence,
    _cosine_similarity,
)


# ================================================================
# 数据结构
# ================================================================

@dataclass
class Atom:
    """MinerU Markdown 解析后的语义原子。"""
    type: str  # heading / text / table / formula_block / formula_inline / list / image
    text: str
    protected: bool = False
    level: int = 0  # 仅 heading 有效，标题层级 1-6


# ================================================================
# 阶段 1: 语义原子解析
# ================================================================

# 受保护块正则：表格 / 块级公式 / 图片（行内公式暂不单独提取，保留在文本中）
_PROTECTED_RE = re.compile(
    r'(?P<formula_block>\$\$[^$]+\$\$)'    # 块级公式 $$...$$
    r'|(?P<table><table>.*?</table>)'       # HTML 表格
    r'|(?P<image>!\[[^\]]*\]\([^)]+\))',    # Markdown 图片
    re.DOTALL,
)

# 行内公式（不作为受保护块切片，而是在文本原子内标记）
_INLINE_FORMULA_RE = re.compile(r'(?<!\$)\$[^$]+\$(?!\$)')

# 标题行
_HEADING_RE = re.compile(r'^#{1,6}\s+(.+)$')

# 列表项
_LIST_BULLET_RE = re.compile(r'^[-\*]\s+')
_LIST_NUMBERED_RE = re.compile(r'^\d+\.\s+')


def _count_heading_level(line: str) -> int:
    """计算标题层级（# 的数量）。"""
    n = 0
    for ch in line:
        if ch == '#':
            n += 1
        else:
            break
    return n


def _is_list_line(line: str) -> bool:
    """判断是否为列表项。"""
    return bool(_LIST_BULLET_RE.match(line) or _LIST_NUMBERED_RE.match(line))


def _parse_text_region(text: str) -> List[Atom]:
    """解析文本区域（受保护块之间的间隙）为原子列表。"""
    atoms: List[Atom] = []
    lines = text.split('\n')
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 空行 → 跳过
        if not stripped:
            i += 1
            continue

        # 标题行
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            level = _count_heading_level(stripped)
            atoms.append(Atom(
                type="heading",
                text=stripped,
                level=min(level, 6),
            ))
            i += 1
            continue

        # 列表行 → 收集连续列表项为一个原子
        if _is_list_line(stripped):
            list_lines = []
            while i < n and lines[i].strip() and _is_list_line(lines[i].strip()):
                list_lines.append(lines[i])
                i += 1
            atoms.append(Atom(type="list", text='\n'.join(list_lines)))
            continue

        # 普通文本段落 → 收集连续非空且非特殊行为一个原子
        para_lines = []
        while i < n and lines[i].strip():
            cur = lines[i].strip()
            if _HEADING_RE.match(cur) or _is_list_line(cur):
                break
            para_lines.append(lines[i])
            i += 1

        if para_lines:
            atoms.append(Atom(type="text", text='\n'.join(para_lines)))

    return atoms


def parse_atoms(markdown_text: str) -> List[Atom]:
    """
    将 MinerU 输出的 Markdown 解析为语义原子列表。

    提取顺序：受保护块（表格/公式/图片）先切出，
    剩余文本区域再按标题/列表/段落分解。
    """
    atoms: List[Atom] = []
    pos = 0

    for m in _PROTECTED_RE.finditer(markdown_text):
        # 当前受保护块之前的文本区域
        gap = markdown_text[pos:m.start()]
        if gap.strip():
            atoms.extend(_parse_text_region(gap))

        # 受保护块本身
        kind = m.lastgroup
        if kind:
            atoms.append(Atom(type=kind, text=m.group(), protected=True))

        pos = m.end()

    # 最后一个受保护块之后的文本
    gap = markdown_text[pos:]
    if gap.strip():
        atoms.extend(_parse_text_region(gap))

    return atoms


# ================================================================
# 阶段 2+3: MarkdownChunker
# ================================================================

class MarkdownChunker(NodeParser):
    """
    MinerU Markdown 语义分块器。

    三阶段流水线：
      1. parse_atoms：解析 Markdown 为语义原子
      2. SentenceSplitter：长文本原子递归切分（受保护原子原样保留）
      3. 相似度合并：相邻节点语义相似且 token 预算内 → 合并

    用法（替代 SentenceSplitter + SimilarityMergeNodeParser）：
        pipeline = IngestionPipeline(
            transformations=[
                MarkdownChunker(max_tokens=512, threshold=0.8),
                Settings.embed_model,
            ],
            ...
        )

    对纯文本文档（无 Markdown 结构）自动降级——全部归类为 text 原子，
    走标准 SentenceSplitter + SimilarityMerge 路径。
    """

    max_tokens: int = 512
    threshold: float = 0.8
    text_chunk_size: int = 256

    def _parse_nodes(
        self, nodes: List[BaseNode], **kwargs
    ) -> List[BaseNode]:
        """主入口：Document/TextNode 列表 → 语义分块后的 TextNode 列表。"""
        all_result_nodes: List[BaseNode] = []

        for node in nodes:
            text = node.text or ""
            if not text.strip():
                continue

            # 阶段 1: 语义原子解析
            atoms = parse_atoms(text)

            # 阶段 2: 文本原子切分
            split_nodes = self._split_atoms(atoms, node)

            # 阶段 3: 相似度合并
            merged = self._similarity_merge(split_nodes)

            all_result_nodes.extend(merged)

        return all_result_nodes

    # ----------------------------------------------------------
    # 阶段 2: 原子 → TextNode
    # ----------------------------------------------------------

    def _split_atoms(
        self, atoms: List[Atom], source_node: BaseNode
    ) -> List[TextNode]:
        """
        将原子列表转换为 TextNode 列表。

        - heading/list/短text：直接转换为单节点
        - 长 text：SentenceSplitter 递归切分
        - table/formula_block/image：原样保留为节点
        """
        splitter = SentenceSplitter(
            chunk_size=self.text_chunk_size,
            chunk_overlap=0,
            paragraph_separator="\n\n\n",
            secondary_chunking_regex=r"[^,.;。？！]+[,.;。？！]?|[,.;。？！]",
        )

        nodes: List[TextNode] = []
        base_metadata = dict(source_node.metadata or {})

        # 标题栈：追踪当前章节路径
        section_stack: List[Tuple[int, str]] = []

        for atom in atoms:
            # 标题 → 更新章节栈
            if atom.type == "heading":
                while (
                    section_stack
                    and section_stack[-1][0] >= atom.level
                ):
                    section_stack.pop()
                heading_text = atom.text.lstrip("#").strip()
                section_stack.append((atom.level, heading_text))

            section_path = (
                " > ".join(h[1] for h in section_stack)
                if section_stack
                else None
            )

            if atom.protected:
                # 受保护原子 → 作为整体节点
                meta = self._build_atom_meta(
                    base_metadata, atom, section_path
                )
                nodes.append(TextNode(text=atom.text, metadata=meta))

            elif bge_token_count(atom.text) <= self.text_chunk_size:
                # 短文本 / 标题 / 列表 → 单节点
                meta = self._build_atom_meta(
                    base_metadata, atom, section_path
                )
                nodes.append(TextNode(text=atom.text, metadata=meta))

            else:
                # 长文本 → SentenceSplitter 递归切分
                split_doc = Document(
                    text=atom.text,
                    metadata={**base_metadata, "atom_type": atom.type},
                )
                sub_nodes = splitter.get_nodes_from_documents([split_doc])
                for sn in sub_nodes:
                    if section_path:
                        sn.metadata["section_path"] = section_path
                nodes.extend(sub_nodes)

        return nodes

    @staticmethod
    def _build_atom_meta(
        base: dict, atom: Atom, section_path: Optional[str]
    ) -> dict:
        """构建单个原子的 metadata。"""
        meta = {**base, "atom_type": atom.type}
        if section_path:
            meta["section_path"] = section_path
        if atom.type == "heading":
            meta["heading_level"] = atom.level
        elif atom.type == "table":
            meta["has_table"] = True
            meta["protected"] = True
        elif atom.type in ("formula_block", "formula_inline"):
            meta["has_formula"] = True
            meta["protected"] = True
        elif atom.type == "image":
            meta["protected"] = True

        return meta

    # ----------------------------------------------------------
    # 阶段 3: 相似度合并
    # ----------------------------------------------------------

    def _similarity_merge(
        self, nodes: List[TextNode]
    ) -> List[TextNode]:
        """
        相似度合并。对相邻节点检查 token 预算和边界语义相似度，
        满足条件则合并。受保护节点同样参与合并流程。

        合并后的节点合并标记（has_table / has_formula / protected）。
        """
        nodes = [n for n in nodes if n.text and n.text.strip()]
        if len(nodes) <= 1:
            return list(nodes)

        nodes = list(nodes)  # 复制，避免原地修改

        if Settings.embed_model is None:
            raise ValueError(
                "Settings.embed_model 未配置，请先调用 configure_settings()"
            )

        embed_model = Settings.embed_model
        merged: List[TextNode] = []
        i = 0

        while i < len(nodes):
            if i == len(nodes) - 1:
                merged.append(nodes[i])
                break

            # token 预算检查
            combined_text = nodes[i].text + nodes[i + 1].text
            if bge_token_count(combined_text) > self.max_tokens:
                merged.append(nodes[i])
                i += 1
                continue

            # 边界相似度
            last_sent = _get_last_sentence(nodes[i].text)
            first_sent = _get_first_sentence(nodes[i + 1].text)

            try:
                emb_a = embed_model.get_text_embedding(last_sent)
                emb_b = embed_model.get_text_embedding(first_sent)
                sim = _cosine_similarity(
                    np.array(emb_a), np.array(emb_b)
                )
            except Exception:
                sim = 0.0

            if sim >= self.threshold:
                merged_meta = self._merge_metadata(
                    nodes[i].metadata, nodes[i + 1].metadata
                )
                nodes[i + 1] = TextNode(
                    text=combined_text, metadata=merged_meta
                )
                i += 1
            else:
                merged.append(nodes[i])
                i += 1

        return merged

    @staticmethod
    def _merge_metadata(left: dict, right: dict) -> dict:
        """合并两个节点的 metadata，左优先，flag 类标记取并集。"""
        merged_meta = {**left}
        for k, v in right.items():
            if k not in merged_meta:
                merged_meta[k] = v

        # 合并 atom_type → atom_types 列表（去重）
        left_type = left.get("atom_type", "")
        right_type = right.get("atom_type", "")
        types: List[str] = []
        if left_type:
            types.append(left_type)
        if right_type and right_type != left_type:
            types.append(right_type)
        if len(types) > 1:
            merged_meta["atom_types"] = types

        # OR 合并布尔标记
        for flag in ("protected", "has_table", "has_formula"):
            if right.get(flag):
                merged_meta[flag] = True

        return merged_meta
