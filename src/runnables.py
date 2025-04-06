from dataclasses import dataclass
from typing import Dict, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from nodes.answer import AnswerInput, createAnswerChain
from nodes.classification import ClassificationInput, createClassificationChain
from nodes.faq_chain import RAGInput, createFAQChain
from nodes.paraphrase import ParaphraseInput, createParaphraseChain
from nodes.rag_chain import RAGInput, createRAGChain
from nodes.summary import SummarizeInput, createSummarizeChain


@dataclass
class SupplierRunnablesVLLM:
    """
    Контейнер для всех Runnable, используемых в SupplierAssistant.

    Атрибуты:
        scenarioChain: Runnable для анализа сценария и определения пути (SQL или ElasticSearch).
        sqlShots: Runnable для подбора примеров SQL-запросов.
        sqlGen: Runnable для генерации SQL-запросов на основе примеров.
        elasticSearch: Runnable для поиска по документам через Elasticsearch.
        answer: Runnable для генерации финального ответа пользователю.
        contextualize_chain: Runnable для генерации генерации контекста ответа пользователю.
    """

    answer: Runnable[AnswerInput, AIMessage]
    rag_chain: Runnable[RAGInput, AIMessage]
    faq_chain: Runnable[RAGInput, AIMessage]
    classification: Runnable[ClassificationInput, AIMessage]
    paraphrase: Runnable[ParaphraseInput, AIMessage]
    summary: Runnable[SummarizeInput, AIMessage]


def createSupplierRunnablesVLLM(
    llm_name: str,
    headers: Optional[Dict[str, str]] = None,
) -> SupplierRunnablesVLLM:
    """
    Создаёт и возвращает набор Runnable для Supplier.

    Args:
        llm_name: Название модели LLM.
        headers: Заголовки для HTTP-запросов (необязательно).

    Returns:
        SupplierRunnablesVLLM: Набор Runnable для SupplierAssistant.
    """
    # Инициализация всех цепочек
    answer = createAnswerChain(llm_name=llm_name, headers=headers)
    rag_chain = createRAGChain()
    faq_chain = createFAQChain()
    paraphrase = createParaphraseChain(llm_name=llm_name, headers=headers)
    classification = createClassificationChain(llm_name=llm_name, headers=headers)
    summary = createSummarizeChain(llm_name=llm_name, headers=headers)
    return SupplierRunnablesVLLM(
        answer=answer,
        rag_chain=rag_chain,
        faq_chain=faq_chain,
        classification=classification,
        paraphrase=paraphrase,
        summary=summary,
    )
