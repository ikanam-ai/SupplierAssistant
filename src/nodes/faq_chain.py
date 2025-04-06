from typing import List, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from pymilvus import Collection, connections

from config import COLLECTIONS, MILVUS_HOST, MILVUS_PORT


class RAGInput(TypedDict):
    query: str
    messages: List[BaseMessage]


class RAGOutput(BaseModel):
    search_results: List[dict] = Field(description="Результаты поиска из Milvus")
    top_k: int = Field(description="Количество возвращённых результатов", default=5)
    collection_stats: dict = Field(description="Статистика по коллекциям", default_factory=dict)


def createFAQChain(
    host: str = MILVUS_HOST,
    port: int = MILVUS_PORT,
) -> Runnable[RAGInput, RAGOutput]:
    """
    Создаёт цепочку для поиска релевантных документов в Milvus с использованием BM25.
    Ищет по всем указанным коллекциям и возвращает топ-5 результатов из объединённых результатов.
    """

    class RAGRunnable(Runnable[RAGInput, RAGOutput]):
        def invoke(self, input_data: RAGInput) -> RAGOutput:
            user_query = input_data["query"]
            connections.connect("default", host=host, port=port)

            all_results = []
            collection_stats = {}

            for collection_name in COLLECTIONS:
                try:
                    collection = Collection(collection_name)
                    collection.load()

                    # Выполняем поиск в текущей коллекции
                    results = collection.search(
                        data=[user_query],
                        anns_field="bm25",
                        param={"metric_type": "BM25"},
                        limit=5,
                        output_fields=["title", "description"],
                    )

                    # Форматируем результаты для текущей коллекции
                    collection_results = []
                    for hit in results[0]:
                        collection_results.append(
                            {
                                "collection": collection_name,
                                "content": hit.fields.get("description", ""),
                                "title": hit.fields.get("title", ""),
                                "score": float(hit.score),
                            }
                        )

                    # Сохраняем статистику по коллекции
                    collection_stats[collection_name] = {
                        "count": len(collection_results),
                        "max_score": max([r["score"] for r in collection_results]) if collection_results else 0,
                    }

                    all_results.extend(collection_results)

                except Exception as e:
                    print(f"Ошибка при работе с коллекцией {collection_name}: {str(e)}")
                    collection_stats[collection_name] = {"error": str(e)}

            # Сортируем все результаты по score (по убыванию)
            all_results_sorted = sorted(all_results, key=lambda x: x["score"], reverse=True)

            # Берем топ-5 результатов из всех коллекций
            top_results = all_results_sorted[:5]

            return RAGOutput(search_results=top_results, top_k=len(top_results), collection_stats=collection_stats)

    return RAGRunnable()
