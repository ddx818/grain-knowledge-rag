"""
MarkdownChunker 单元测试。

覆盖三种场景：
  1. 含表格的 Markdown → 表格完整保留
  2. 含公式的 Markdown → 公式完整保留
  3. 纯文本文档 → 降级行为正确
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from src.markdown_chunker import (
    MarkdownChunker,
    parse_atoms,
    Atom,
)
from src.chunker import bge_token_count


# ================================================================
# 阶段 1 测试: parse_atoms
# ================================================================

class TestParseAtoms:
    """测试语义原子解析。"""

    def test_plain_text_only(self):
        """纯文本 → 全部归类为 text 原子。"""
        text = "入库小麦水分应严格控制在安全水分标准以内。\n\n当小麦水分超过安全标准时，应及时通风。"
        atoms = parse_atoms(text)
        types = [a.type for a in atoms]
        assert types == ["text", "text"]

    def test_headings(self):
        """标题行被正确识别。"""
        text = "### 3.1 水分控制\n\n入库小麦水分应严格控制。"
        atoms = parse_atoms(text)
        assert atoms[0].type == "heading"
        assert atoms[0].level == 3
        assert atoms[1].type == "text"

    def test_table_protected(self):
        """HTML 表格被标记为受保护原子。"""
        text = "各等级标准如下：\n\n<table>\n<tr><td>等级</td><td>水分</td></tr>\n</table>\n\n入库时应检测。"
        atoms = parse_atoms(text)
        assert atoms[1].type == "table"
        assert atoms[1].protected is True
        # 表格内容完整
        assert "<table>" in atoms[1].text
        assert "</table>" in atoms[1].text

    def test_formula_block_protected(self):
        """块级公式被标记为受保护原子。"""
        text = "结露温度计算公式：\n\n$$T_d = \\frac{b \\times \\gamma(T)}{a + \\gamma(T)}$$\n\n当仓壁温度低于露点温度时应采取保温措施。"
        atoms = parse_atoms(text)
        assert atoms[1].type == "formula_block"
        assert atoms[1].protected is True
        # 公式前后文本各为独立原子
        types = [a.type for a in atoms]
        assert "text" in types

    def test_mixed_document(self):
        """混合文档：标题 + 文本 + 表格 + 公式。"""
        text = (
            "## 第二章 储藏标准\n\n"
            "入库小麦应符合以下标准。\n\n"
            "<table><tr><td>等级</td><td>水分(%)</td></tr></table>\n\n"
            "结露计算公式：\n\n"
            "$$T_d = T - \\frac{100 - RH}{5}$$\n\n"
            "以上标准必须严格执行。"
        )
        atoms = parse_atoms(text)
        types = [(a.type, a.protected) for a in atoms]
        assert ("heading", False) in types
        assert ("table", True) in types
        assert ("formula_block", True) in types
        assert ("text", False) in types

    def test_list_items(self):
        """列表项被收集为单个原子。"""
        text = "- 一等小麦：≤12.5%\n- 二等小麦：≤12.5%\n- 三等小麦：≤12.5%"
        atoms = parse_atoms(text)
        assert atoms[0].type == "list"
        assert len(atoms[0].text.split('\n')) == 3

    def test_image_protected(self):
        """图片被标记为受保护原子。"""
        text = "粮仓剖面图：\n\n![图1 粮仓剖面](images/warehouse.png)\n\n如图所示。"
        atoms = parse_atoms(text)
        assert atoms[1].type == "image"
        assert atoms[1].protected is True


# ================================================================
# 阶段 2+3 测试: MarkdownChunker
# ================================================================

class TestMarkdownChunker:
    """测试 MarkdownChunker 完整流水线。"""

    @pytest.fixture(autouse=True)
    def setup_embed_model(self):
        """确保 Settings.embed_model 已配置。"""
        from src.settings import configure_settings
        configure_settings()

    # --- 表格保护 ---

    def test_table_not_split(self):
        """表格作为整体节点，不被 SentenceSplitter 切分。"""
        chunker = MarkdownChunker(max_tokens=512, threshold=0.8)
        doc = Document(
            text="标准如下：\n\n<table><tr><td>等级</td><td>水分</td></tr></table>\n\n以上。",
            metadata={"file_name": "test.pdf"},
        )
        nodes = chunker._parse_nodes([doc])

        table_nodes = [n for n in nodes if n.metadata.get("has_table")]
        assert len(table_nodes) >= 1
        for tn in table_nodes:
            assert "<table>" in tn.text
            assert "</table>" in tn.text

    def test_table_with_low_threshold_merges(self):
        """低阈值下表格与上下文合并。"""
        chunker = MarkdownChunker(max_tokens=512, threshold=0.0)  # 阈值 0 强制合并
        doc = Document(
            text="各等级小麦的安全水分标准如下：\n\n"
                 "<table><tr><td>一等</td><td>≤12.5</td></tr></table>",
            metadata={"file_name": "test.pdf"},
        )
        nodes = chunker._parse_nodes([doc])
        # 低阈值下应合并为一个节点
        assert len(nodes) == 1
        assert "安全水分标准" in nodes[0].text
        assert "<table>" in nodes[0].text

    def test_table_token_budget_enforced(self):
        """token 预算超限时不合并。"""
        chunker = MarkdownChunker(max_tokens=30, threshold=0.998)  # 极小预算 + 极高阈值
        # 构造一个短表格 + 长前后文
        doc = Document(
            text="A. " * 200 + "\n\n<table><tr><td>x</td></tr></table>\n\n" + "B. " * 200,
            metadata={"file_name": "test.pdf"},
        )
        nodes = chunker._parse_nodes([doc])
        table_nodes = [n for n in nodes if "<table>" in n.text]
        assert len(table_nodes) >= 1

    # --- 公式保护 ---

    def test_formula_not_split(self):
        """公式作为整体节点，不被切分。"""
        chunker = MarkdownChunker(max_tokens=512, threshold=0.8)
        doc = Document(
            text="结露温度计算公式：\n\n$$T_d = T - \\frac{100 - RH}{5}$$\n\n"
                 "当仓壁温度低于露点温度时应采取保温措施。",
            metadata={"file_name": "test.pdf"},
        )
        nodes = chunker._parse_nodes([doc])

        formula_nodes = [n for n in nodes if n.metadata.get("has_formula")]
        assert len(formula_nodes) >= 1
        for fn in formula_nodes:
            assert "$$" in fn.text

    # --- 纯文本降级 ---

    def test_plain_text_graceful_degradation(self):
        """纯文本文档降级为类 SentenceSplitter 行为。"""
        chunker = MarkdownChunker(max_tokens=512, threshold=0.8)
        long_text = (
            "入库小麦水分应严格控制在安全水分标准以内。各等级小麦的安全水分标准"
            "参见相关国家标准。入库时应进行水分检测，确保符合标准要求。"
            "当小麦水分超过安全标准时，应及时进行机械通风干燥处理。"
        ) * 20
        doc = Document(text=long_text, metadata={"file_name": "test.pdf"})
        nodes = chunker._parse_nodes([doc])

        # 应产出多个 chunk
        assert len(nodes) >= 2
        # 所有 chunk 应 ≤ max_tokens
        for node in nodes:
            assert bge_token_count(node.text) <= 512
        # 不应有任何 protected 标记
        for node in nodes:
            assert not node.metadata.get("protected")

    # --- metadata ---

    def test_section_path_in_metadata(self):
        """章节路径被正确写入 metadata（标题自身的 path 不含子标题，内容节点包含完整路径）。"""
        chunker = MarkdownChunker(max_tokens=512, threshold=0.8)
        doc = Document(
            text="## 第三章 储藏技术\n\n### 3.1 水分控制\n\n"
                 "入库小麦水分应严格控制。",
            metadata={"file_name": "test.pdf"},
        )
        nodes = chunker._parse_nodes([doc])

        # 顶级标题不包含子标题
        assert "第三章" in nodes[0].metadata.get("section_path", "")
        assert "3.1 水分控制" not in nodes[0].metadata.get("section_path", "")

        # 子标题和内容节点包含完整路径
        for node in nodes[1:]:
            sp = node.metadata.get("section_path", "")
            assert "第三章" in sp, f"Expected path containing 第三章, got: {sp}"
            assert "3.1 水分控制" in sp, f"Expected path containing 3.1 水分控制, got: {sp}"

    def test_file_metadata_preserved(self):
        """源文档 metadata 被保留在所有 chunk 中。"""
        chunker = MarkdownChunker(max_tokens=512, threshold=0.8)
        doc = Document(
            text="入库小麦水分应严格控制。",
            metadata={"file_name": "标准.pdf", "file_type": "pdf"},
        )
        nodes = chunker._parse_nodes([doc])

        for node in nodes:
            assert node.metadata.get("file_name") == "标准.pdf"
            assert node.metadata.get("file_type") == "pdf"
