"""提示词和配置加载器。从外部文件读取，支持版本切换。"""
import json
import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_system_prompt(version: str = "default") -> str:
    """加载系统提示词。version='v2' 则读取 system_v2.txt。"""
    suffix = f"_{version}" if version != "default" else ""
    path = PROMPTS_DIR / f"system{suffix}.txt"

    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {path}")

    return path.read_text(encoding="utf-8").strip()


def load_agent_config() -> Dict[str, Any]:
    """加载 Agent 模型配置。"""
    path = PROMPTS_DIR / "agent_config.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_model_kwargs() -> Dict[str, Any]:
    """从配置文件提取模型参数。环境变量优先级高于配置文件。"""
    config = load_agent_config()
    model_cfg = config.get("model", {})

    return {
        "model": os.getenv("DEEPSEEK_MODEL", model_cfg.get("name", "deepseek-chat")),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", model_cfg.get("base_url", "https://api.deepseek.com/v1")),
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", model_cfg.get("temperature", 0.3))),
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", model_cfg.get("max_tokens", 1024))),
    }
