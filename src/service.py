"""
知识库服务层 —— IngestionPipeline 管道化入库 + 混合检索。
"""
from pathlib import Path
from typing import List, Dict, Any
import os

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.schema import TextNode, BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.loader import load_all_documents, load_incremental_documents, _load_single_file
from src.markdown_chunker import MarkdownChunker
from src.state_manager import IngestionState, compute_file_hash


class KnowledgeBaseService:
    """知识库服务，管理文档全生命周期。"""

    def __init__(
        self,
        project_root: str = None,
        collection_name: str = "grain_storage",
    ):
        if project_root is None:
            project_root = str(Path(__file__).resolve().parent.parent)

        self.project_root = Path(project_root)
        abs_chroma = str(self.project_root / "chroma_data")
        self.chroma_dir = os.path.relpath(abs_chroma, os.getcwd()) if os.path.isabs(abs_chroma) else abs_chroma
        self.docs_dir = str(self.project_root / "documents")
        self.collection_name = collection_name
        self.state_path = str(self.project_root / "chroma_data" / "ingestion_state.json")
        self._client = None

    # ============================================================
    # 内部组件
    # ============================================================

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.chroma_dir)
        return self._client

    def _get_vector_store(self, overwrite: bool = False):
        client = self._get_client()
        if overwrite:
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
        collection = client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return ChromaVectorStore(chroma_collection=collection)

    def _get_state(self) -> IngestionState:
        return IngestionState(self.state_path)

    # ============================================================
    # 文档上传（保留 SHA256 去重）
    # ============================================================

    def upload_file(self, source_path: str) -> Dict[str, Any]:
        source = Path(source_path).resolve()
        if not source.exists():
            return {"dest": None, "action": "error", "message": f"文件不存在: {source_path}"}

        suffix = source.suffix.lower()
        if suffix not in (".pdf", ".docx"):
            return {"dest": None, "action": "error", "message": f"不支持的类型: {suffix}"}

        import shutil
        docs_dir = self.project_root / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        dest = docs_dir / source.name
        rel_name = str(dest.relative_to(docs_dir))

        source_hash = compute_file_hash(source)
        state = self._get_state()
        existing = state.find_by_hash(source_hash)
        if existing is not None:
            return {"dest": existing, "action": "skip", "message": f"内容已存在于: {existing}"}

        action = "update" if dest.exists() else "new"
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)

        state.remove_file(rel_name)
        state.save()

        return {
            "dest": str(dest.relative_to(self.project_root)),
            "action": action,
            "message": "已更新，待入库" if action == "update" else "上传成功",
        }

    def upload_files(self, source_paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        for path in source_paths:
            result = self.upload_file(path)
            result["source"] = path
            results.append(result)
        return results

    # ============================================================
    # 文档入库 —— IngestionPipeline 管道化
    # ============================================================

    def _delete_chunks_by_filepath(self, file_path: str):
        """从 ChromaDB 中删除指定文件的所有旧 chunk。"""
        import logging
        logger = logging.getLogger("uvicorn")
        try:
            collection = self._get_client().get_collection(self.collection_name)
            collection.delete(where={"file_path": file_path})
        except Exception as e:
            logger.warning(f"ChromaDB 删除旧 chunk 失败 ({file_path}): {e}")

    def _ingest_documents(self, documents: list, full_rebuild: bool = False) -> int:
        """核心入库流水线：MarkdownChunker（原子解析+递归切分+相似度合并）→ embedding → ChromaDB。"""
        import logging
        import errno
        logger = logging.getLogger("uvicorn")
        from src.settings import configure_settings
        configure_settings()

        vector_store = self._get_vector_store(overwrite=full_rebuild)

        pipeline = IngestionPipeline(
            transformations=[
                MarkdownChunker(max_tokens=512, threshold=0.8),
                Settings.embed_model,
            ],
            vector_store=vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

        try:
            nodes = pipeline.run(documents=documents)
        except OSError as e:
            if e.errno == errno.ENOSPC:
                msg = "磁盘空间不足，无法写入 ChromaDB 索引文件。请清理磁盘后重试。"
                logger.error(msg)
                raise RuntimeError(msg) from e
            raise
        finally:
            self._client = None  # 重置客户端，下次重连
        return len(nodes)

    def ingest_file(self, file_path: str) -> int:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        doc = _load_single_file(path)
        if doc is None:
            raise ValueError(f"无法加载文件: {path.name}（可能是扫描版 PDF）")

        # 删除旧 chunk 后重新入库（IngestionPipeline 不跨批次追踪）
        self._delete_chunks_by_filepath(str(path))

        n = self._ingest_documents([doc])

        file_hash = compute_file_hash(path)
        rel_path = str(path.relative_to(self.project_root / "documents"))
        state = self._get_state()
        state.set_file_hash(rel_path, file_hash)
        state.save()

        return n

    def ingest_files(self, file_paths: List[str]) -> Dict[str, Any]:
        total_chunks = 0
        results = []
        for fp in file_paths:
            try:
                n = self.ingest_file(fp)
                results.append({"file": fp, "status": "ok", "chunks": n})
                total_chunks += n
            except Exception as e:
                results.append({"file": fp, "status": "error", "message": str(e)})
        return {"total_files": len(file_paths), "total_chunks": total_chunks, "details": results}

    def ingest_all(self, full_rebuild: bool = False) -> int:
        if full_rebuild:
            documents = load_all_documents()
        else:
            state = self._get_state()
            documents, state = load_incremental_documents(state=state)
            if len(documents) == 0:
                return 0

        total = self._ingest_documents(documents, full_rebuild=full_rebuild)

        state = self._get_state()
        if full_rebuild:
            state.state["files"] = {}
            docs_dir = self.project_root / "documents"
            if docs_dir.exists():
                for fp in docs_dir.rglob("*"):
                    if fp.is_file() and fp.suffix.lower() in (".pdf", ".docx"):
                        rel_path = str(fp.relative_to(docs_dir))
                        state.set_file_hash(rel_path, compute_file_hash(fp))
        state.save()

        return total

    # ============================================================
    # 检索
    # ============================================================

    def _get_all_nodes(self) -> List[BaseNode]:
        import logging
        logger = logging.getLogger("uvicorn")
        try:
            collection = self._get_client().get_collection(self.collection_name)
            data = collection.get()
        except Exception as e:
            logger.error(f"ChromaDB 读取 collection 失败: {e}")
            return []
        nodes = []
        for i, (text, meta) in enumerate(zip(data["documents"] or [], data["metadatas"] or [])):
            meta = meta or {}
            node_id = (data["ids"] or [None])[i] if i < len(data["ids"] or []) else None
            nodes.append(TextNode(text=text, metadata=meta, node_id=node_id))
        return nodes

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        import logging
        logger = logging.getLogger("uvicorn")
        try:
            vector_store = self._get_vector_store(overwrite=False)
            index = VectorStoreIndex.from_vector_store(vector_store)
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
        return [
            {
                "score": round(node.score, 4) if node.score else 0,
                "text": node.text,
                "file_name": node.metadata.get("file_name", ""),
                "file_type": node.metadata.get("file_type", ""),
                "page_count": node.metadata.get("page_count", 0),
            }
            for node in nodes
        ]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        import logging
        logger = logging.getLogger("uvicorn")
        from src.retriever import HybridRetriever

        try:
            vector_store = self._get_vector_store(overwrite=False)
            nodes = self._get_all_nodes()
            hr = HybridRetriever(vector_store, nodes=nodes, top_k=top_k)
            return hr.search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            # 降级：如果向量检索路也失败，返回空
            return []

    # ============================================================
    # 状态查询
    # ============================================================

    def _check_chromadb_health(self) -> dict:
        """自检 ChromaDB 是否可正常读写。"""
        import logging
        logger = logging.getLogger("uvicorn")
        try:
            client = self._get_client()
            collection = client.get_collection(self.collection_name)
            count = collection.count()
            # 尝试一次轻量查询验证索引完整性
            _ = collection.get(limit=1)
            logger.debug(f"ChromaDB 健康检查通过，共 {count} 条记录")
            return {"status": "healthy", "total_chunks": count}
        except Exception as e:
            logger.error(f"ChromaDB 健康检查失败: {e}")
            return {"status": "degraded", "error": str(e), "total_chunks": 0}

    @staticmethod
    def _get_disk_usage(path: str) -> dict:
        """获取指定路径的磁盘使用情况。"""
        import shutil
        try:
            usage = shutil.disk_usage(path)
            free_gb = round(usage.free / (1024 ** 3), 1)
            total_gb = round(usage.total / (1024 ** 3), 1)
            used_pct = round((1 - usage.free / usage.total) * 100, 1)
            return {"free_gb": free_gb, "total_gb": total_gb, "used_pct": used_pct}
        except Exception:
            return {}

    def get_status(self) -> Dict[str, Any]:
        state = self._get_state()
        health = self._check_chromadb_health()
        disk = self._get_disk_usage(self.chroma_dir)
        return {
            "collection_name": self.collection_name,
            "total_chunks": health["total_chunks"],
            "chromadb": health["status"],
            "chromadb_error": health.get("error"),
            "tracked_files": len(state.state.get("files", {})),
            "last_save": state.state.get("stats", {}).get("last_save"),
            "chroma_dir": self.chroma_dir,
            "docs_dir": self.docs_dir,
            "disk": disk,
        }
