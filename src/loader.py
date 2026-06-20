"""
文档加载器：SimpleDirectoryReader + 扫描版 PDF OCR fallback。
保留 SHA256 增量加载逻辑，函数签名不变。
"""
from pathlib import Path
from typing import Optional, Set
import numpy as np

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader, DocxReader

from src.state_manager import IngestionState, compute_file_hash

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 扫描版 PDF OCR fallback
# ============================================================

_OCR = None


def _get_ocr():
    """PaddleOCR 模块级单例，避免每次 OCR 调用都重新加载模型。"""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(lang='ch', use_angle_cls=True, show_log=False)
    return _OCR


def _ocr_pdf_pages(file_path: Path) -> str:
    """对扫描版 PDF 逐页 OCR，返回拼接后的全文。"""
    import fitz

    ocr = _get_ocr()
    all_texts = []

    with fitz.open(str(file_path)) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]

            try:
                result = ocr.ocr(img_array, cls=True)
                if result and result[0]:
                    page_text = "\n".join(
                        line[1][0] for line in result[0] if line[1][1] > 0.5
                    )
                    all_texts.append(page_text)
                else:
                    all_texts.append("")
            except Exception as e:
                print(f"  [WARN] OCR 第 {page_num + 1} 页失败: {e}")
                all_texts.append("")

    return "\n".join(all_texts)


# ============================================================
# file_extractor 配置
# ============================================================

def _make_pdf_extractor():
    """返回 PDF extractor：先提取文本层，扫描版走 OCR。"""
    pdf_reader = PyMuPDFReader()

    def _extract(file_path, **kwargs):
        page_docs = pdf_reader.load_data(file_path)
        full_text = "\n".join([p.text for p in page_docs])

        if len(full_text.strip()) < 100:
            ocr_text = _ocr_pdf_pages(file_path)
            if len(ocr_text.strip()) < 50:
                return []
            return [Document(
                text=ocr_text,
                metadata={
                    "file_name": Path(file_path).name,
                    "file_path": str(file_path),
                    "file_type": "pdf",
                    "page_count": len(page_docs),
                    "ocr": True,
                },
            )]

        return [Document(
            text=full_text,
            metadata={
                "file_name": Path(file_path).name,
                "file_path": str(file_path),
                "file_type": "pdf",
                "page_count": len(page_docs),
            },
        )]

    return _extract


def _make_docx_extractor():
    """返回 DOCX extractor。"""
    docx_reader = DocxReader()

    def _extract(file_path, **kwargs):
        page_docs = docx_reader.load_data(file_path)
        full_text = "\n".join([p.text for p in page_docs])
        return [Document(
            text=full_text,
            metadata={
                "file_name": Path(file_path).name,
                "file_path": str(file_path),
                "file_type": "docx",
                "page_count": len(page_docs),
            },
        )]

    return _extract


# ============================================================
# SHA256 增量过滤
# ============================================================

def _filter_incremental(
    docs: list,
    state: IngestionState,
    base_dir: Path,
    existing_paths: set,
):
    """用 SHA256 过滤未变更文件。返回 (新文档列表, 跳过数量)。"""
    filtered = []
    skipped = 0

    for doc in docs:
        fp = Path(doc.metadata.get("file_path", ""))
        if not fp.exists():
            continue

        rel_path = str(fp.relative_to(base_dir))
        existing_paths.add(rel_path)

        file_hash = compute_file_hash(fp)
        stored_hash = state.get_stored_hash(rel_path)
        if stored_hash == file_hash:
            skipped += 1
            continue

        state.set_file_hash(rel_path, file_hash)
        filtered.append(doc)

    state.cleanup_deleted(existing_paths)
    return filtered, skipped


# ============================================================
# 公开 API（签名不变）
# ============================================================

def load_all_documents(docs_dir: str = "documents") -> list:
    """加载 documents/ 下所有 PDF 和 DOCX 文件。"""
    base = _PROJECT_ROOT / docs_dir
    reader = SimpleDirectoryReader(
        input_dir=str(base),
        file_extractor={
            ".pdf": _make_pdf_extractor(),
            ".docx": _make_docx_extractor(),
        },
        recursive=True,
        filename_as_id=True,
    )
    all_docs = reader.load_data()

    # 过滤扫描版 PDF
    scanned = [d for d in all_docs if d.metadata.get("ocr")]
    for d in scanned:
        print(f"  [OCR] {d.metadata.get('file_name', '?')}")

    print(f"全量加载: {len(all_docs)} 个文档")
    return all_docs


def load_incremental_documents(
    docs_dir: str = "documents",
    state: Optional[IngestionState] = None,
):
    """增量加载：跳过 SHA256 未变更文件。返回 (新文档列表, 更新后的状态管理器)。"""
    base = _PROJECT_ROOT / docs_dir

    if state is None:
        state = IngestionState(
            str(_PROJECT_ROOT / "chroma_data" / "ingestion_state.json")
        )

    reader = SimpleDirectoryReader(
        input_dir=str(base),
        file_extractor={
            ".pdf": _make_pdf_extractor(),
            ".docx": _make_docx_extractor(),
        },
        recursive=True,
        filename_as_id=True,
    )

    all_docs = reader.load_data()

    existing_paths: Set[str] = set()
    new_docs, skipped = _filter_incremental(all_docs, state, base, existing_paths)

    print(f"增量加载: 新增/变更 {len(new_docs)} 个, 跳过 {skipped} 个 (未变更)")
    return new_docs, state


def _load_single_file(file_path: Path) -> Optional[Document]:
    """加载单个文件为 Document。扫描版 PDF 走 OCR。"""
    suffix = file_path.suffix.lower()
    if suffix not in (".pdf", ".docx"):
        return None

    try:
        if suffix == ".pdf":
            extractor = _make_pdf_extractor()
            docs = extractor(str(file_path))
            return docs[0] if docs else None
        else:
            extractor = _make_docx_extractor()
            docs = extractor(str(file_path))
            return docs[0] if docs else None
    except Exception as e:
        print(f"[ERROR] 加载 {file_path.name} 失败: {e}")
        return None
