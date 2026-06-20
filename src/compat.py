"""
兼容性补丁模块

ragas 0.4.3 依赖 `langchain_community.chat_models.vertexai.ChatVertexAI`，
但 langchain-community >= 0.4.0 已将该类移除，迁移至独立包 langchain-google-vertexai。

此模块在首次导入时自动修补 sys.modules，使 ragas 能够正常导入。
任何需要导入 ragas 的代码，请先 `import compat`。
"""

import sys
import importlib
from types import ModuleType


def _install_compat() -> None:
    """安装兼容性补丁：将 langchain_community.chat_models.vertexai 映射到 langchain_google_vertexai"""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        try:
            from langchain_google_vertexai import ChatVertexAI  # noqa: F401
        except ImportError:
            return  # langchain-google-vertexai 未安装，跳过补丁

        # 创建一个虚拟模块并注入 ChatVertexAI
        spec = importlib.util.find_spec("langchain_google_vertexai")
        compat_module = importlib.util.module_from_spec(spec) if spec else None
        if compat_module is None:
            compat_module = ModuleType(module_name)

        compat_module.__name__ = module_name
        compat_module.ChatVertexAI = ChatVertexAI
        sys.modules[module_name] = compat_module


_install_compat()
