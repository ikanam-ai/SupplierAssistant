import logging
from datetime import datetime
from pymongo import MongoClient
from typing import List, TypedDict, cast
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from runnables import AcademicRunnablesOllama
from config import MONGO_DB_PATH

class State(TypedDict):
    messages: List[BaseMessage]
    query: str
    final_output: str
    user_id: str
    search_results_faq: List[dict]
    search_results_rag: List[dict]
    classification_results: str
    image_data: str

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AcademicAssistant")

class AcademicAssistant:
    def __init__(
        self,
        academic_runnables: AcademicRunnablesOllama,
        checkpointer: MongoDBSaver,
    ) -> None:
        self.academic_runnables = academic_runnables
        self.mongo_client = MongoClient(MONGO_DB_PATH)
        self.db = self.mongo_client['checkpointing_db']
        self.logs_collection = self.db["logs"]

        graph_builder = StateGraph(State)
        
        # Добавляем ноды
        graph_builder.add_node("classification", self.classification)
        graph_builder.add_node("rag_search", self.rag_search)
        graph_builder.add_node("fag_search", self.fag_search)
        graph_builder.add_node("answer", self.answer)

        # Настраиваем граф
        graph_builder.set_entry_point("classification")
        graph_builder.add_edge("classification", "fag_search")
        graph_builder.add_edge("fag_search", "rag_search")
        graph_builder.add_edge("rag_search", "answer")
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
        log_entry = {
            "user_id": user_id,
            "action": action,
            "details": details,
            "timestamp": datetime.now()
        }
        try:
            self.logs_collection.insert_one(log_entry)
            logger.info(f"Log saved: {action} for user {user_id}")
        except Exception as e:
            logger.error(f"Log save failed: {e}")

    def classification(self, state: State, config: RunnableConfig) -> State:
        result = self.academic_runnables.classification.invoke({
            "query": state["query"],
            "messages": state.get("messages", []),
            "user_id": state["user_id"]
        })
        self._save_flat_log(
            state["user_id"],
            "query",
            state["query"],
            state["query"]
        )    
        details = f"Классификация: {result.classification}"
        self._save_flat_log(
            state["user_id"],
            "classification",
            state["query"],
            details
        )
        
        return {**state, "classification_results": result.classification}

    def fag_search(self, state: State, config: RunnableConfig) -> State:
        result = self.academic_runnables.faq_chain.invoke({
            "query": state["query"],
            "messages": state.get("messages", []),
            "user_id": state["user_id"]
        })
        
        details = f"Найдено FAQ документов: {len(result.search_results)}"
        self._save_flat_log(
            state["user_id"],
            "fag_search",
            state["query"],
            details
        )
        
        return {**state, "search_results_faq": result.search_results}

    def rag_search(self, state: State, config: RunnableConfig) -> State:
        result = self.academic_runnables.rag_chain.invoke({
            "query": state["query"],
            "messages": state.get("messages", []),
            "user_id": state["user_id"]
        })
        
        details = f"Найдено RAG документов: {len(result.search_results)}"
        self._save_flat_log(
            state["user_id"],
            "rag_search",
            state["query"],
            details
        )
        
        return {**state, "search_results_rag": result.search_results, "image_data": result.image_data}

    def answer(self, state: State, config: RunnableConfig) -> State:
        context = "\n".join([
            *[f"FAQ: {doc['content'][:200]}..." for doc in state.get("search_results_faq", [])],
            *[f"RAG: {doc['content'][:200]}..." for doc in state.get("search_results_rag", [])]
        ])
        
        answer_result = self.academic_runnables.answer.invoke({
            "context": context,
            "query": state["query"],
        })

        full_output = f"{answer_result.final_output}\n\nClassification: {state['classification_results']}"
        
        self._save_flat_log(
            state["user_id"],
            "answer",
            state["query"],
            f"Ответ: {full_output[:500]}..."
        )
        
        return {
            **state,
            "final_output": full_output,
            "messages": state.get("messages", []) + [
                HumanMessage(content=state["query"]),
                AIMessage(content=full_output)
            ]
        }