"""模型下载脚本。

运行此脚本从 HuggingFace Hub 下载所需模型：
  - BAAI/bge-large-zh-v1.5    → 嵌入模型（用于向量化）
  - BAAI/bge-reranker-v2-m3   → Cross-Encoder 精排模型

使用方法：
  uv run python src/download.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# 项目根目录（相对于此脚本向上两级：src/download.py → 项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "src"

MODELS = [
    {
        "repo_id": "BAAI/bge-large-zh-v1.5",
        "local_dir": str(MODELS_DIR / "bge-large-zh-v1.5"),
        "description": "BGE-large-zh 嵌入模型",
    },
    {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "local_dir": str(MODELS_DIR / "bge-reranker-v2-m3"),
        "description": "Cross-Encoder 精排模型",
    },
]


def download_model(repo_id: str, local_dir: str, description: str) -> bool:
    """下载单个模型，返回是否成功。"""
    if Path(local_dir).exists() and any(Path(local_dir).iterdir()):
        print(f"[跳过] {description} — 已存在: {local_dir}")
        return True

    print(f"[下载] {description} ({repo_id}) → {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
        )
        print(f"[完成] {description}")
        return True
    except Exception as e:
        print(f"[失败] {description}: {e}", file=sys.stderr)
        return False


def main() -> int:
    print("=" * 60)
    print("  粮食仓储 RAG 助手 — 模型下载")
    print("=" * 60)
    print(f"  模型目录: {MODELS_DIR}")
    print()

    failed = []
    for model in MODELS:
        if not download_model(**model):
            failed.append(model["description"])

    print()
    if failed:
        print(f"!! {len(failed)} 个模型下载失败: {', '.join(failed)}")
        print("  请检查网络连接后重试。")
        return 1
    else:
        print("[OK] 所有模型就绪！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
