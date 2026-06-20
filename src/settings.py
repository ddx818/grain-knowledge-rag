"""
全局 Settings 配置 —— LlamaIndex 依赖注入入口。

项目启动时调用 configure_settings() 一次，所有模块通过 Settings.xxx 获取组件，
不再手动跨模块传递 embed_model / llm / node_parser。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT / "src" / "bge-large-zh-v1.5")

# 统一加载 .env
load_dotenv(PROJECT_ROOT / ".env")

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SentenceSplitter


_configured = False


def configure_settings():
    """配置全局 Settings，幂等（多次调用仅首次生效）。"""
    global _configured
    if _configured:
        return

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=MODEL_DIR,
        max_length=512,
        device="cpu",
        local_files_only=True,
    )

    Settings.llm = OpenAILike(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "1024")),
        is_chat_model=True,
    )

    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=0)
    Settings.chunk_size = 256
    Settings.chunk_overlap = 0

    _configured = True
