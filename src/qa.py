"""
RAG 问答模块：检索上下文 + DeepSeek LLM 生成答案。

用法：
    from src.qa import QAService
    qa = QAService(kb_service)
    answer = qa.ask("粮食安全水分标准是多少？")
"""
from src.settings import configure_settings
from llama_index.core import Settings


class QAService:
    """
    RAG 问答服务：混合检索 + LLM 生成。

    流程：
        用户提问 → 混合检索(top_k=5) → 拼 Prompt → DeepSeek 生成 → 返回答案
    """

    def __init__(self, kb_service):
        """
        kb_service: KnowledgeBaseService 实例
        """
        self.kb = kb_service

        configure_settings()
        self._llm = Settings.llm

    @staticmethod
    def _build_prompt(query: str, contexts: list) -> str:
        """将检索到的上下文拼接成 Prompt。"""
        ctx_blocks = []
        for i, ctx in enumerate(contexts):
            src = ctx.get("file_name", "未知")
            ctx_blocks.append(
                f"[参考{i+1} | 来源: {src}]\n{ctx['text']}"
            )
        context_str = "\n\n".join(ctx_blocks)

        return f"""你是一个粮食仓储知识专家助手。请根据以下参考资料回答用户的问题。

## 参考资料
{context_str}

## 要求
- 如果参考资料中有相关信息，请基于资料给出准确的回答
- 如果参考资料中没有相关信息，请明确说"根据现有资料，无法回答该问题"
- 回答时请注明引用的资料来源
- 回答简洁清晰，不要编造资料中没有的内容

## 用户问题
{query}

## 回答
"""

    def ask(self, query: str, top_k: int = 5) -> dict:
        """
        RAG 问答。

        返回：
            {"query": "原问题", "answer": "LLM 回答", "sources": [...], "contexts": [...]}
        """
        # 1. 混合检索
        contexts = self.kb.hybrid_search(query, top_k=top_k)

        if not contexts:
            return {
                "query": query,
                "answer": "知识库中暂无相关信息。",
                "sources": [],
                "contexts": [],
            }

        # 2. 拼 Prompt
        prompt = self._build_prompt(query, contexts)

        # 3. 调用 LLM
        response = self._llm.complete(prompt)

        # 4. 提取来源
        sources = list(set(
            c["file_name"] for c in contexts
        ))

        return {
            "query": query,
            "answer": response.text,
            "sources": sources,
            "contexts": contexts,
        }
