from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from config import PROGRAMS_TABLE_PATH
from nodes.answer import AnswerInput, createAnswerChain
from nodes.rag_chain import RAGInput, createRAGChain
from nodes.faq_chain import RAGInput, createFAQChain
from nodes.classification import ClassificationInput, createClassificationChain


@dataclass
class AcademicRunnablesOllama:
    """
    Контейнер для всех Runnable, используемых в AcademicAssistant.

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


def createAcademicRunnablesOllama(
    llm_name: str,
    headers: Optional[Dict[str, str]] = None,
) -> AcademicRunnablesOllama:
    """
    Создаёт и возвращает набор Runnable для Academic.

    Args:
        llm_name: Название модели LLM.
        headers: Заголовки для HTTP-запросов (необязательно).

    Returns:
        AcademicRunnablesOllama: Набор Runnable для AcademicAssistant.
    """
    # Инициализация всех цепочек
    # FIXME: LOOK CAREFULLY ON THIS PIPELINE BECAUSE THERE ARE A LOF OF CHANGES
    answer = createAnswerChain(llm_name=llm_name, headers=headers)
    rag_chain = createRAGChain()
    faq_chain = createFAQChain()
    classification = createClassificationChain(llm_name=llm_name, headers=headers)
    return AcademicRunnablesOllama(
        answer=answer,
        rag_chain=rag_chain,
        faq_chain=faq_chain,
        classification=classification
    )
