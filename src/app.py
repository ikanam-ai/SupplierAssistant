import asyncio
import logging
import uuid

import streamlit as st

from config import APP_LLM_NAME, MONGO_DB_PATH
from handler import AcademicHandler, AcademicOptions

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("AcademicStreamlitBot")


def academic_handler():
    """
    Создаёт и возвращает обработчик для AcademicAssistant.
    """
    options = AcademicOptions(
        llm_name=APP_LLM_NAME,
        psycopg_checkpointer=MONGO_DB_PATH,
    )
    return AcademicHandler(options)


handler = academic_handler()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


async def get_response(prompt: str) -> str:
    """
    Асинхронно обрабатывает запрос пользователя и возвращает ответ.
    """
    user_id = st.session_state.user_id
    try:
        result, _ = await handler.ahandle_prompt(prompt, user_id)
        return result
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        return "❌ Произошла ошибка при обработке запроса. Пожалуйста, попробуйте ещё раз."


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Чат-бот для абитуриентов РАНХиГС")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Что у вас на уме?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(get_response(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)
