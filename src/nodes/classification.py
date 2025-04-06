from typing import Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from openai import OpenAI
from pydantic import BaseModel, Field

from config import CLIENT_URL, LLM_NAME


class ClassificationInput(TypedDict):
    query: str
    messages: List[BaseMessage]


ClassificationType = Literal["термин", "проблема", "работа", "оператор", "нейтрально"]


class ClassificationOutput(BaseModel):
    classification: ClassificationType = Field(description="Тип классификации запроса")


def createClassificationChain(
    llm_name: str = LLM_NAME,
    headers: Optional[Dict[str, str]] = None,
) -> Runnable[ClassificationInput, ClassificationOutput]:
    """
    Создаёт цепочку классификации запросов на 5 типов:
    - термин: вопросы о определениях и понятиях
    - проблема: вопросы об ошибках и проблемах
    - работа: вопросы о документах пользователя
    - оператор: запросы к человеку-оператору
    - нейтрально: простые сообщения
    """
    prompt_template = PromptTemplate.from_template(
        """Классифицируйте пользовательский запрос ровно в одну из 5 категорий:
        - "термин": вопросы о определениях, терминах, понятиях
        - "проблема": вопросы об ошибках, неполадках, проблемах
        - "работа": вопросы о документах или работе пользователя
        - "оператор": запросы, требующие человека-оператора
        - "нейтрально": простые сообщения
        
        Верните ТОЛЬКО слово категории, ничего больше.
        
        Примеры:
        "Что такое квантовый компьютер?" -> термин
        "Я получаю ошибку 404 при загрузке" -> проблема
        "Проверьте мою научную статью" -> работа
        "Соедините меня с поддержкой" -> оператор
        "Привет, как дела?" -> нейтрально
        "Добрый день!" -> нейтрально
        
        Запрос для классификации: {query}
        Категория:"""
    )

    client = OpenAI(api_key="EMPTY", base_url=CLIENT_URL)

    class ClassificationRunnable(Runnable[ClassificationInput, ClassificationOutput]):
        def invoke(self, input_data: ClassificationInput) -> ClassificationOutput:
            query = input_data["query"].strip().lower()

            prompt = prompt_template.format(query=query)

            response = client.chat.completions.create(
                model=llm_name,
                messages=[
                    {"role": "system", "content": "Вы помощник для классификации запросов."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
                stream=False,
            )

            result = response.choices[0].message.content.strip().lower()

            # Валидация результата
            valid_types = {"термин", "проблема", "работа", "оператор", "нейтрально"}
            if result not in valid_types:
                result = "оператор"  # значение по умолчанию

            return ClassificationOutput(classification=result)

    return ClassificationRunnable()
