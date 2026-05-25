"""
RagasEvaluator —— LLM-as-a-Judge 评估器。

封装 Ragas 框架，通过 DeepSeek 评判 LLM 对 RAG 管线进行语义级评估。

用法：
    from eval.ragas_evaluator import RagasEvaluator
    evaluator = RagasEvaluator()
    results = evaluator.evaluate_all(dataset)
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.compat  # noqa: F401 — ragas/langchain-community 兼容补丁，必须在 ragas 之前导入

from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("user_input", "retrieved_contexts", "reference")


class RagasEvaluator:
    """LLM-as-a-Judge 评估器，支持检索和生成两阶段独立评估。"""

    def __init__(self, llm=None, embed_model=None):
        """
        llm: Ragas 评判 LLM（LangchainLLMWrapper 实例），不传则自动创建 DeepSeek
        embed_model: Ragas embedding 模型，不传则自动加载 BGE-large-zh
        """
        self._llm = llm if llm is not None else self._create_judge_llm()
        self._embed_model = embed_model if embed_model is not None else self._create_embed_model()

    @staticmethod
    def _create_judge_llm():
        """创建 DeepSeek 评判 LLM。"""
        import os
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI

        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请在 .env 文件中配置")

        lc_llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.0,  # 评判任务用低温，保证评分稳定性
        )
        return LangchainLLMWrapper(lc_llm)

    @staticmethod
    def _create_embed_model():
        """加载本地 BGE-large-zh embedding 模型。"""
        model_dir = str(PROJECT_ROOT / "src" / "bge-large-zh-v1.5")
        return HuggingFaceEmbeddings(
            model_name=model_dir,
            model_kwargs={"device": "cpu"},
        )

    @staticmethod
    def _build_dataset(data: list[dict]) -> dict:
        """将收集的数据列表转换为 Ragas evaluate() 所需的 dict 格式。"""
        if not data:
            raise ValueError("评估数据集为空")
        if not isinstance(data, list) or not isinstance(data[0], dict):
            raise ValueError("评估数据格式错误：应为 list[dict]")
        for key in _REQUIRED_KEYS:
            if key not in data[0]:
                raise ValueError(f"评估数据缺少必需字段: {key}")

        return {
            "user_input": [item["user_input"] for item in data],
            "response": [item.get("response", "") for item in data],
            "retrieved_contexts": [item.get("retrieved_contexts", []) for item in data],
            "reference": [item.get("reference", "") for item in data],
        }

    @staticmethod
    def _extract_scores(result) -> dict[str, list[float]]:
        """从 Ragas EvaluationResult 中提取各指标的分数列表。"""
        import numpy as np

        df = result.to_pandas()
        scores = {}
        for col in df.columns:
            series = df[col]
            nan_mask = series.isna()
            if nan_mask.any():
                nan_indices = series.index[nan_mask].tolist()
                logger.warning("指标 %s 在第 %s 行存在 NaN，已排除", col, nan_indices)
            series = series.dropna()
            if len(series) == 0:
                logger.warning("指标 %s 所有值均为 NaN", col)
                continue
            if series.apply(lambda x: isinstance(x, (int, float))).all():
                scores[col] = series.tolist()

        if not scores:
            logger.warning("_extract_scores 返回空字典——所有指标列均为 NaN 或非数值类型")

        return scores

    def evaluate_retrieval(self, data: list[dict]) -> dict[str, list[float]]:
        """评估检索质量：ContextPrecision, ContextRecall。"""
        dataset = self._build_dataset(data)
        metrics = [ContextPrecision(), ContextRecall()]
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
        )
        return self._extract_scores(result)

    def evaluate_generation(self, data: list[dict]) -> dict[str, list[float]]:
        """评估生成质量：Faithfulness, AnswerRelevancy, FactualCorrectness。"""
        dataset = self._build_dataset(data)
        metrics = [Faithfulness(), AnswerRelevancy(), FactualCorrectness()]
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embed_model,
        )
        return self._extract_scores(result)

    def evaluate_all(self, data: list[dict]) -> dict[str, list[float]]:
        """一次执行全量指标评估。"""
        retrieval_scores = self.evaluate_retrieval(data)
        generation_scores = self.evaluate_generation(data)
        return {**retrieval_scores, **generation_scores}
