
<p align="center">
  <img src="extensions/images/logo.png" alt="SupplierAssistant Logo" width="500"/>
</p>


# 🧾 SupplierAssistant — интеллектуальный бот для Портала поставщиков



**SupplierAssistant** — Telegram-бот на базе LLM (Large Language Model), разработанный для ответов на вопросы пользователей, связанные с информацией о **Портале поставщиков** — интернет-ресурсе, предназначенном для автоматизации деятельности заказчиков и поставщиков, оперативного заключения сделок и повышения прозрачности контрактных отношений.

---

## 🚀 Возможности

- 🤖 Ответы на естественном языке по структуре, функциям и правилам работы Портала поставщиков  
- 📄 Объяснение процедур закупок, участия в тендерах и заключения контрактов  
- 🏢 Поддержка как для заказчиков, так и для поставщиков  
- 📌 Навигация по нормативной базе, регламентам и техническим требованиям  
- 🔍 Использование RAG (retrieval-augmented generation) для точных и актуальных ответов  

---

## 💡 Пример диалога

> **Пользователь**: Как пройти аккредитацию на Портале поставщиков?  
> **SupplierAssistant**: Чтобы пройти аккредитацию, вам необходимо зарегистрироваться на портале, заполнить данные компании, загрузить учредительные документы и дождаться проверки модератора. Подробности в разделе «Регистрация и аккредитация».

---

## 🛠️ Технологии

- [Python 3.12.2](https://www.python.org/)
- [Poetry](https://python-poetry.org/)
- LLM (OpenAI / Mistral / Llama)
- Retrieval-Augmented Generation (RAG)

---

## 📦 Установка

```bash
git clone https://github.com/your-org/SupplierAssistant.git
cd SupplierAssistant
poetry install
```

---

## 🐳 Запуск через Docker

### 1. Соберите Docker-образ:

```bash
docker build -t supplier-assistant .
```

### 2. Запустите контейнер:

```bash
docker run -d \
  --name supplier-bot \
  --gpus all \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  supplier-assistant
```

## 📞 Обратная связь

Если вы нашли ошибку или хотите предложить улучшение — создайте issue или отправьте pull request.  

Для сотрудничества и внедрения в корпоративную инфраструктуру — свяжитесь с разработчиком.
