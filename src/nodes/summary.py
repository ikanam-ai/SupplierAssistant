from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from openai import OpenAI
from pydantic import BaseModel, Field

from config import CLIENT_URL, LLM_NAME


class SummarizeInput(TypedDict):
    text: str
    messages: List[BaseMessage]


class SummarizeOutput(BaseModel):
    summary: str = Field(description="Суммаризированный текст (до 300 символов)")


def createSummarizeChain(
    llm_name: str = LLM_NAME,
    headers: Optional[Dict[str, str]] = None,
) -> Runnable[SummarizeInput, SummarizeOutput]:
    """
    Создает цепочку для краткой суммаризации текста (до 300 символов)
    с сохранением ключевых терминов, цифр, дат и реквизитов.
    """
    prompt_template = PromptTemplate.from_template(
        """Сократите текст до 300 символов, сохраняя:
        - Технические термины и аббревиатуры
        - Номера документов и ссылки
        - Ключевые цифры и даты
        - Основную суть
        
        Текст: {text}
        Краткое изложение:"""
    )

    client = OpenAI(api_key="EMPTY", base_url=CLIENT_URL)

    class SummarizeRunnable(Runnable[SummarizeInput, SummarizeOutput]):
        def invoke(self, input_data: SummarizeInput) -> SummarizeOutput:
            response = client.chat.completions.create(
                model=llm_name,
                messages=[{"role": "user", "content": prompt_template.format(text=input_data["text"])}],
                temperature=0.1,
                max_tokens=100,
            )
            summary = response.choices[0].message.content.strip()
            return SummarizeOutput(summary=summary)

    return SummarizeRunnable()
