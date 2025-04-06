from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from openai import OpenAI
from pydantic import BaseModel, Field

from config import CLIENT_URL, LLM_NAME


class ParaphraseInput(TypedDict):
    query: str
    messages: List[BaseMessage]


class ParaphraseOutput(BaseModel):
    paraphrased_query: str = Field(description="Перефразированный запрос")


def createParaphraseChain(
    llm_name: str = LLM_NAME,
    headers: Optional[Dict[str, str]] = None,
) -> Runnable[ParaphraseInput, ParaphraseOutput]:
    """
    Создаёт цепочку для умного перефраза запросов:
    - Исправляет опечатки и грамматические ошибки
    - Обобщает частные вопросы
    - Делает запросы более формальными
    - Обезличивает личные данные
    - Сохраняет термины в оригинальном виде
    - Не изменяет корректные формальные запросы
    """
    prompt_template = PromptTemplate.from_template(
        """Перефразируйте запрос в сфере госзакупок, соблюдая правила:
        1. Корректно используйте термины: 44-ФЗ, 223-ФЗ, ЕИС, ЭТП, НМЦК и др.
        2. Формализуйте согласно нормам закупочной документации
        3. Удалите коммерчески чувствительную информацию
        4. Сохраните юридические формулировки без изменений
        5. Используйте официальный стиль извещений
        6. Упростите сложные формулировки без потери смысла
        
        Примеры:
        "как подать заявку на тендер?" -> "Как подать заявку на участие в закупке?"
        "У меня проблема с аккредитацией на росэлторг" -> "Возникла проблема с аккредитацией на ЭТП"
        "Что такое НМЦК в 44-ФЗ?" -> "Что такое начальная (максимальная) цена контракта согласно 44-ФЗ?"
        "Моя компания не прошла отбор в тендере №123" -> "Участник не прошел отбор в закупке"
        
        Верните ТОЛЬКО перефразированный запрос, ничего больше.
        ДАЖЕ ЕСЛИ НЕ ОТНОСИТСЯ К ТЕМЕ ГОСЗАКУПОК ТО СДЕЛАЙ ЧТОБЫ ОТНОСИЛОСЬ В КОНТЕКСТЕ ПРОДАЖИ И ПОКУПКИ
        
        Запрос для перефраза: {query}
        Перефразированный запрос:"""
    )

    client = OpenAI(api_key="EMPTY", base_url=CLIENT_URL)

    class ParaphraseRunnable(Runnable[ParaphraseInput, ParaphraseOutput]):
        def invoke(self, input_data: ParaphraseInput) -> ParaphraseOutput:
            original_query = input_data["query"].strip()

            prompt = prompt_template.format(query=original_query)

            response = client.chat.completions.create(
                model=llm_name,
                messages=[
                    {"role": "system", "content": "Вы помощник для перефраза запросов."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=100,
                stream=False,
            )

            paraphrased_query = response.choices[0].message.content.strip()

            print("paraphrased_query")
            print(paraphrased_query)
            print("paraphrased_query")

            return ParaphraseOutput(paraphrased_query=paraphrased_query)

    return ParaphraseRunnable()
