import asyncio
import logging

from telegram import KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import PROGRAMS_TABLE_PATH, TOKEN
from handler import AcademicHandler, AcademicOptions

# Настройка логирования
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("AcademicBot")


import pandas as pd
from telegram import KeyboardButton, ReplyKeyboardMarkup

from config import APP_LLM_NAME, MONGO_DB_PATH


# Загрузка данных из Excel-файла
def load_programs_data():
    df = pd.read_excel(PROGRAMS_TABLE_PATH)
    return df


# Получение уникальных кластеров
def get_clusters(df):
    return df["cluster"].unique().tolist()


# Получение программ по кластеру
def get_programs_by_cluster(df, cluster):
    return df[df["cluster"] == cluster]["program"].tolist()


# Инициализация данных
df = load_programs_data()
clusters = get_clusters(df)


def academic_handler() -> AcademicHandler:
    """
    Создаёт и возвращает обработчик для AcademicAssistant.

    Returns:
        AcademicHandler: Обработчик запросов.
    """
    options = AcademicOptions(
        llm_name=APP_LLM_NAME,
        psycopg_checkpointer=MONGO_DB_PATH,
    )
    handler = AcademicHandler(options)
    return handler


# Инициализация обработчика
handler = academic_handler()


# Основная клавиатура
def get_main_keyboard():
    keyboard = [
        [KeyboardButton("🔄 Новый диалог")],
        [KeyboardButton("📚 Выбрать кластер")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


# Обработчик команды /start
async def start(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    logger.info(f"Команда /start получена от пользователя {user_id}")

    welcome_message = """
    Здравствуйте! 👋 Я ваш нейропомощник из Президентской академии. Готов помочь вам с выбором и сравнением образовательных программ, а также предоставить подробную информацию о каждой из них. Просто напишите, какая программа вас интересует, — и я постараюсь ответить максимально быстро и подробно!
    """

    await update.message.reply_text(welcome_message, parse_mode="Markdown", reply_markup=get_main_keyboard())


# Обработчик текстовых сообщенийasync
async def handle_message(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    user_input = update.message.text
    logger.info(f"Сообщение от пользователя {user_id}: {user_input}")

    if user_input == "🔄 Новый диалог":
        result, _ = await handler.ahandle_prompt("начать новый диалог", str(user_id))
        processing_message = await update.message.reply_text("⏳ Обрабатываю ваш запрос...", parse_mode="Markdown")
        welcome_message = """
        Здравствуйте! 👋 Я ваш нейропомощник из Президентской академии. Готов помочь вам с выбором и сравнением образовательных программ, а также предоставить подробную информацию о каждой из них. Просто напишите, какая программа вас интересует, — и я постараюсь ответить максимально быстро и подробно!
        """
        await processing_message.edit_text(welcome_message, parse_mode="Markdown")
        return

    elif user_input == "📚 Выбрать кластер":
        # Создаем клавиатуру с кластерами
        cluster_keyboard = [[KeyboardButton(cluster)] for cluster in clusters]
        reply_markup = ReplyKeyboardMarkup(cluster_keyboard, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text("Выберите кластер:", reply_markup=reply_markup)
        return

    elif user_input in clusters:
        # Получаем программы по выбранному кластеру
        programs = get_programs_by_cluster(df, user_input)
        programs = list(set(programs))
        programs_message = "\n".join([f"• {program}" for program in programs])
        await update.message.reply_text(
            f"Вы можете задать вопросы по данным программам или выбрать 2 для сравнения \n'{user_input}':\n{programs_message}",
            reply_markup=get_main_keyboard(),  # Возвращаем основную клавиатуру
        )
        return

    # Обработка обычных запросов
    processing_message = await update.message.reply_text("⏳ Обрабатываю ваш запрос...", parse_mode="Markdown")
    try:
        result, _ = await handler.ahandle_prompt(user_input, str(user_id))
        # Отправляем новое сообщение с результатом и основной клавиатурой
        await update.message.reply_text(result, parse_mode="Markdown", reply_markup=get_main_keyboard())
        # Удаляем сообщение "Обрабатываю ваш запрос..."
        await processing_message.delete()
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text(
            "К сожалению, по вашему запросу информации не найдено. Пожалуйста, попробуйте переформулировать вопрос или свяжитесь с Приёмной комиссией РАНХиГС для получения подробной консультации: pkranepa@ranepa.ru, +7 (499) 956-99-99. Мы всегда рады помочь вам!",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard(),
        )
        await processing_message.delete()


# Запуск бота
async def run_bot():
    logger.info("Запуск бота...")
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await application.initialize()
    await application.start()
    logger.info("Бот запущен и готов к получению сообщений.")
    await application.updater.start_polling()
    logger.info("Polling запущен.")
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.info("Цикл событий уже запущен. Создание задачи для бота.")
            loop.create_task(run_bot())
            loop.run_forever()
        else:
            logger.info("Запуск нового цикла событий.")
            loop.run_until_complete(run_bot())
    except RuntimeError as e:
        if str(e) == "This event loop is already running":
            logger.info("Цикл событий уже запущен. Создание задачи для бота.")
            loop = asyncio.get_running_loop()
            loop.create_task(run_bot())
            loop.run_forever()
        else:
            raise
