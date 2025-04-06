from typing import Dict, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from openai import OpenAI

from config import ANSWER_NODE_LLM_TEMPERATURE, ANSWER_NODE_SYSTEM_PROMPT, LLM_NAME, CLIENT_BASE_URL

class AnswerInput(TypedDict):
    context: str
    query: str

class AnswerOutput(BaseModel):
    final_output: str = Field(description="Финальный ответ пользователю")

def createAnswerChain(
    llm_name: str = LLM_NAME,
    headers: Optional[Dict[str, str]] = None,
    temperature: float = ANSWER_NODE_LLM_TEMPERATURE,
) -> Runnable[AnswerInput, AnswerOutput]:
    """
    Создаёт цепочку генерации ответа, используя OpenAI-совместимый API (например, YandexGPT через локальный сервер).
    """

    prompt_template = PromptTemplate.from_template(
        """
        {system_prompt}

        Контекст:
        {context}

        Запрос пользователя:
        {query}

        Сформулируй полный и информативный ответ на основе предоставленного контекста и запроса пользователя.
        Не указывай, откуда именно получен контекст.
        """
    )

    client = OpenAI(
        api_key="EMPTY",
        base_url=CLIENT_BASE_URL
    )

    class AnswerRunnable(Runnable[AnswerInput, AnswerOutput]):
        def invoke(self, input_data: AnswerInput) -> AnswerOutput:
            prompt = prompt_template.format(
                system_prompt=ANSWER_NODE_SYSTEM_PROMPT,
                query=input_data["query"],
                context=input_data["context"]
            )
        
            
            response = client.chat.completions.create(
                model=llm_name,
                messages=[
                    {"role": "system", "content": ANSWER_NODE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                # temperature=temperature,
                stream=False
            )

            result = response.choices[0].message.content.strip()
            return AnswerOutput(final_output=result)

    return AnswerRunnable()

