"""
入库状态管理器。

记录每个文件的 SHA256 哈希，下次入库时自动跳过未变更的文件。
状态文件存储在 chroma_data/ingestion_state.json。
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class IngestionState:
    """管理文件级增量入库状态。"""

    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self.state: Dict = {"files": {}, "stats": {}}
        self._load()

    def _load(self):
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["stats"]["last_save"] = datetime.now().isoformat()
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_stored_hash(self, file_path: str) -> Optional[str]:
        """获取某文件上次入库时的哈希值。"""
        return self.state["files"].get(file_path)

    def set_file_hash(self, file_path: str, file_hash: str):
        """更新某文件的哈希记录。"""
        self.state["files"][file_path] = file_hash

    def remove_file(self, file_path: str):
        """删除某文件的记录（文件已从磁盘删除时调用）。"""
        self.state["files"].pop(file_path, None)

    def find_by_hash(self, file_hash: str) -> Optional[str]:
        """
        反向查找：通过哈希值找到对应的文件名。
        用于判断上传的文件内容是否已存在于知识库（可能以不同文件名存储）。
        """
        for fname, h in self.state["files"].items():
            if h == file_hash:
                return fname
        return None

    def cleanup_deleted(self, existing_paths: set):
        """清理状态中已不存在于磁盘的文件记录。"""
        deleted = set(self.state["files"].keys()) - existing_paths
        for path in deleted:
            self.remove_file(path)
        return list(deleted)


def compute_file_hash(file_path: Path) -> str:
    """计算文件的 SHA256 哈希，用于检测文件是否变更。"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(128 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_shorthand(file_hash: str) -> str:
    """返回哈希的前 12 位短写，用于日志展示。"""
    return file_hash[:12]
