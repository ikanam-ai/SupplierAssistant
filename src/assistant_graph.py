import logging
from datetime import datetime
from typing import List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from pymongo import MongoClient

from config import MONGO_DB_PATH
from runnables import SupplierRunnablesVLLM


class State(TypedDict):
    messages: List[BaseMessage]
    query: str
    original_query: str  # Новое поле для хранения оригинального запроса
    final_output: str
    user_id: str
    search_results_faq: List[dict]
    search_results_rag: List[dict]
    classification_results: str
    # image_data: str # FIXME: fix images part
    combined_text: str


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SupplierAssistant")


class SupplierAssistant:
    def __init__(
        self,
        Supplier_runnables: SupplierRunnablesVLLM,
        checkpointer: MongoDBSaver,
    ) -> None:
        self.Supplier_runnables = Supplier_runnables
        self.mongo_client = MongoClient(MONGO_DB_PATH)
        self.db = self.mongo_client["checkpointing_db"]
        self.logs_collection = self.db["logs"]

        graph_builder = StateGraph(State)

        # Добавляем ноды
        graph_builder.add_node("paraphrase", self.paraphrase)
        graph_builder.add_node("classification", self.classification)
        graph_builder.add_node("rag_search", self.rag_search)
        graph_builder.add_node("fag_search", self.fag_search)
        graph_builder.add_node("summary", self.summary)
        graph_builder.add_node("answer", self.answer)

        # Настраиваем граф
        graph_builder.set_entry_point("paraphrase")
        graph_builder.add_edge("paraphrase", "classification")
        graph_builder.add_edge("classification", "fag_search")
        graph_builder.add_edge("fag_search", "rag_search")
        graph_builder.add_edge("rag_search", "summary")
        graph_builder.add_edge("summary", "answer")
        graph_builder.add_edge("answer", END)

        self.graph = graph_builder.compile(checkpointer=checkpointer)

    def _save_flat_log(self, user_id: str, action: str, query: str, details: str):
        """
        Сохраняет лог в MongoDB в плоском табличном формате

        Args:
            user_id: ID пользователя
            action: Название действия/ноды
            query: Текст запроса
            details: Текстовые детали выполнения
        """
        log_entry = {"user_id": user_id, "action": action, "details": details, "timestamp": datetime.now()}
        try:
            self.logs_collection.insert_one(log_entry)
            logger.info(f"Log saved: {action} for user {user_id}")
        except Exception as e:
            logger.error(f"Log save failed: {e}")

    def paraphrase(self, state: State, config: RunnableConfig) -> State:
        """
        Нода для перефразирования запроса:
        - Исправляет опечатки и грамматические ошибки
        - Обобщает частные вопросы
        - Делает запросы более формальными
        - Обезличивает личные данные
        - Сохраняет термины без изменений
        """
        original_query = state["query"]

        # Выполняем перефраз
        paraphrase_result = self.Supplier_runnables.paraphrase.invoke(
            {"query": original_query, "messages": state.get("messages", [])}
        )

        paraphrased_query = paraphrase_result.paraphrased_query

        # Логируем результат
        log_details = f"Перефразировано: {original_query} -> {paraphrased_query}"
        self._save_flat_log(state["user_id"], "paraphrase", original_query, log_details)

        # Обновляем состояние с перефразированным запросом
        return {
            **state,
            "query": paraphrased_query,
            "original_query": original_query,  # Сохраняем оригинальный запрос в состоянии
        }

    def classification(self, state: State, config: RunnableConfig) -> State:
        result = self.Supplier_runnables.classification.invoke(
            {"query": state["query"], "messages": state.get("messages", []), "user_id": state["user_id"]}
        )
        self._save_flat_log(state["user_id"], "query", state["query"], state["query"])
        details = f"Классификация: {result.classification}"
        self._save_flat_log(state["user_id"], "classification", state["query"], details)

        return {**state, "classification_results": result.classification}

    def fag_search(self, state: State, config: RunnableConfig) -> State:
        result = self.Supplier_runnables.faq_chain.invoke(
            {"query": state["query"], "messages": state.get("messages", []), "user_id": state["user_id"]}
        )

        details = f"Найдено FAQ документов: {len(result.search_results)}"
        self._save_flat_log(state["user_id"], "fag_search", state["query"], details)

        return {**state, "search_results_faq": result.search_results}

    def rag_search(self, state: State, config: RunnableConfig) -> State:
        result = self.Supplier_runnables.rag_chain.invoke(
            {"query": state["query"], "messages": state.get("messages", []), "user_id": state["user_id"]}
        )

        details = f"Найдено RAG документов: {len(result.search_results)}"
        self._save_flat_log(state["user_id"], "rag_search", state["query"], details)

        return {**state, "search_results_rag": result.search_results}

    def summary(self, state: State, config: RunnableConfig) -> State:
        """
        Нода для суммаризации найденных чанков:
        - Проверяет каждый чанк отдельно
        - Суммаризирует чанки длиннее 1500 символов
        - Объединяет все чанки (оригинальные и суммаризированные)
        - Сохраняет ключевые термины, цифры, реквизиты
        """
        processed_chunks = []

        # Обрабатываем FAQ чанки
        for doc in state.get("search_results_faq", []):
            content = doc["content"]
            processed_chunks.append(f"{content[:300]}")

        # Обрабатываем RAG чанки
        for doc in state.get("search_results_rag", []):
            content = doc["content"]
            if len(content) > 600:
                # Суммаризируем длинные чанки
                summary_result = self.Supplier_runnables.summary.invoke(
                    {
                        "text": "Запрос пользователя: " + state["query"] + "Контекст: " + content[:2000],
                        "messages": state.get("messages", []),
                    }
                )
                processed_chunks.append(f"{summary_result.summary}")
            else:
                processed_chunks.append(f"{content[:250]}")

        # Объединяем все обработанные чанки
        combined_text = "\n\n".join(processed_chunks)

        # Логируем результат
        original_count = len(state.get("search_results_faq", [])) + len(state.get("search_results_rag", []))
        summarized_count = sum(1 for chunk in processed_chunks if "суммаризировано" in chunk)

        log_details = (
            f"Чанков: {original_count}, "
            f"Суммаризировано: {summarized_count}, "
            f"Итоговый размер: {len(combined_text)} символов"
        )

        self._save_flat_log(state["user_id"], "summary", state["query"], log_details)

        # Обновляем состояние
        return {**state, "combined_text": combined_text, "was_summarized": summarized_count > 0}

    def answer(self, state: State, config: RunnableConfig) -> State:
        """
        Нода для формирования ответа на основе суммаризированных данных
        """
        # Используем уже обработанный текст из состояния
        context = state.get("combined_text", "")

        answer_result = self.Supplier_runnables.answer.invoke(
            {
                "context": context,
                "query": state["query"] + "\nЗапрос пользователя до перефраза:\n" + state["original_query"],
            }
        )

        full_output = f"{answer_result.final_output}\n\nClassification: {state['classification_results']}"

        # Логируем финальный ответ
        self._save_flat_log(
            state["user_id"],
            "answer",
            state["query"],
            f"Ответ на основе {'суммаризированных' if state.get('was_summarized', False) else 'полных'} данных",
        )

        return {
            **state,
            "final_output": full_output,
            "messages": state.get("messages", [])
            + [HumanMessage(content=state["query"]), AIMessage(content=full_output)],
        }
