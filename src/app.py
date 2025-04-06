import io
import logging
import wave

import chainlit as cl
import numpy as np
import torch
import whisper

from config import APP_LLM_NAME, MONGO_DB_PATH
from handler import SupplierHandler, SupplierOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = whisper.load_model("turbo", device=DEVICE)
print(f"Whisper инициализирован на {DEVICE}")


def Supplier_handler() -> SupplierHandler:
    options = SupplierOptions(
        llm_name=APP_LLM_NAME,
        psycopg_checkpointer=MONGO_DB_PATH,
    )
    return SupplierHandler(options)


handler = Supplier_handler()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="🧠 Идеи для ассистента",
            message="Мы участвуем в Tender Hack. Подскажи идеи для ИИ-ассистента на Портале поставщиков — что он должен уметь и как выделиться среди других команд?",
        ),
        cl.Starter(
            label="🗂️ Классификация запросов",
            message="Как реализовать определение категории пользовательского запроса на Портале поставщиков (например, жалоба, вопрос о функционале, техподдержка)?",
        ),
        cl.Starter(
            label="📄 Питч-дек от Icon.Tech",
            message="Помоги составить структуру презентации от Icon.Tech для защиты на Tender Hack. Что должно быть на слайдах?",
        ),
        cl.Starter(
            label="🤖 Бот: ответы и база знаний",
            message="Как связать ИИ-ассистента с базой знаний (инструкциями, FAQ, законами)? Напиши пример архитектуры или пайплайна.",
        ),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("audio_chunks", [])


@cl.on_message
async def main(message: cl.Message):
    user_id = cl.user_session.get("id") or "default_user"
    user_input = message.content
    cl.user_session.set("last_user_input", user_input)  # сохраняем последний ввод
    logger.info(f"📨 Текстовый ввод: {user_input}")

    msg = cl.Message(content="")
    await msg.send()

    result, _ = await handler.ahandle_prompt(user_input, str(user_id))

    # FIXME: fix images part
    # image_paths = [img["image_path"] for img in image_data]
    # unique_figures = [img["number_text"] for img in image_data]

    # image_elements = []
    # for fig_id, img in zip(unique_figures, image_paths):
    #     image_path = os.path.join(IMAGES_PATH, img)
    #     if os.path.isfile(image_path):
    #         image_elements.append(
    #             cl.Image(name=fig_id, path=image_path)
    #         )
    #     else:
    #         logger.warning(f"⚠️ Изображение не найдено: {image_path}")

    await msg.stream_token(result)
    await msg.update()

    # if image_elements:
    #     await cl.Message(
    #         content=", ".join(unique_figures),
    #         elements=image_elements
    #     ).send()

    # Добавляем кнопки для оценки (не обязательные)
    action_message = cl.AskActionMessage(
        content="🤔 Насколько полезен был ответ? (Можно пропустить)",
        actions=[
            cl.Action(name="mark", label="😡 1", payload={"value": 1}),
            cl.Action(name="mark", label="🙁 2", payload={"value": 2}),
            cl.Action(name="mark", label="😐 3", payload={"value": 3}),
            cl.Action(name="mark", label="🙂 4", payload={"value": 4}),
            cl.Action(name="mark", label="😃 5", payload={"value": 5}),
            cl.Action(name="next_question", label="Задать следующий вопрос", payload={"value": "next"}),
        ],
    )
    response = await action_message.send()

    if response and "payload" in response:
        value = response["payload"].get("value")
        if value != "next":
            handler.assistant._save_flat_log(user_id, "rating", value, value)
            await cl.Message(content="✅ Спасибо за вашу оценку!").send()

        else:
            # Переход к следующему вопросу, если не оценили
            await cl.Message(content="Готов к следующему вопросу!").send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    chunks = cl.user_session.get("audio_chunks")
    audio_data = np.frombuffer(chunk.data, dtype=np.int16)
    chunks.append(audio_data)


@cl.on_audio_end
async def on_audio_end():
    audio_chunks = cl.user_session.get("audio_chunks")

    if not audio_chunks:
        await cl.Message(content="🔇 Аудио слишком короткое").send()
        return

    audio_data = np.concatenate(audio_chunks)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_data.tobytes())
    wav_buffer.seek(0)

    with open("temp_audio.wav", "wb") as f:
        f.write(wav_buffer.read())

    result = WHISPER_MODEL.transcribe("temp_audio.wav")
    transcription = result["text"]
    logger.info(f"🗣️ Распознан текст: {transcription}")

    await cl.Message(
        author="Вы",
        type="user_message",
        content=transcription,
        elements=[
            cl.Audio(content=wav_buffer.getvalue(), mime="audio/wav"),
        ],
    ).send()

    await main(cl.Message(content=transcription))
