import json
from typing import List, TypedDict
from pymilvus import connections, Collection
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from config import HOST, PORT, EMBEDDING_MODEL_NAME, E5_COLLECTIONS

class RAGInput(TypedDict):
    query: str
    messages: List[BaseMessage]

class RAGOutput(BaseModel):
    search_results: List[dict] = Field(description="Результаты поиска из Milvus")
    top_k: int = Field(description="Количество возвращённых результатов", default=5)
    collection_stats: dict = Field(description="Статистика по коллекциям", default_factory=dict)
    image_data: list = Field(description="Информация о картинках")

def createRAGChain(
    host: str = HOST, 
    port: int = PORT,
    device: str = "cuda:0"
) -> Runnable[RAGInput, RAGOutput]:
    """
    Создаёт цепочку для поиска релевантных документов в Milvus с использованием E5 эмбеддингов.
    """
    # Инициализация модели для эмбеддингов
    model_name = EMBEDDING_MODEL_NAME
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )
    
    class RAGRunnable(Runnable[RAGInput, RAGOutput]):
        def invoke(self, input_data: RAGInput) -> RAGOutput:
            user_query = input_data["query"]
            
            # Создаем эмбеддинг для запроса
            query_embedding = hf_embeddings.embed_query(user_query)
            
            connections.connect("default", host=host, port=port)
            
            all_results = []
            collection_stats = {}
            
            for collection_name in E5_COLLECTIONS:
                try:
                    collection = Collection(collection_name)
                    collection.load()
                    
                    # Выполняем поиск по эмбеддингу
                    results = collection.search(
                        data=[query_embedding],
                        anns_field="e5",
                        param={"metric_type": "COSINE", "params": {}},
                        limit=5,
                        output_fields=["document_name", "header", "text", "pictures"]
                    )                    
                    # Форматируем результаты
                    collection_results = []
                    for hit in results[0]:
                        collection_results.append({
                            "collection": collection_name,
                            "document_name": hit.fields.get("document_name", ""),
                            "header": hit.fields.get("header", ""),
                            "content": hit.fields.get("text", ""),
                            "pictures": hit.fields.get("pictures", ""),
                            "score": float(hit.score)
                        })
                    
                    # Сохраняем статистику
                    collection_stats[collection_name] = {
                        "count": len(collection_results),
                        "max_score": max([r["score"] for r in collection_results]) if collection_results else 0
                    }
                    
                    all_results.extend(collection_results)
                    
                except Exception as e:
                    print(f"Ошибка при работе с коллекцией {collection_name}: {str(e)}")
                    collection_stats[collection_name] = {"error": str(e)}
            
            # Сортируем и выбираем топ-5
            top_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:5]
            
            print(top_results)
            image_data = top_results[0]["pictures"]
            image_data = json.loads(image_data.replace("'", "\""))

            return RAGOutput(
                search_results=top_results,
                top_k=len(top_results),
                collection_stats=collection_stats,
                image_data=image_data
            )

    return RAGRunnable()