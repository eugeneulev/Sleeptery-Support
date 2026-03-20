# Sleeptery Support Assistant

AI-помощник для поддержки с RAG на базе ChromaDB и LiteLLM.

## Запуск

```bash
# 1. Установить зависимости
uv sync

# 2. Скопировать и заполнить .env
cp .env.example .env

# 3. Запустить сервер
uv run uvicorn main:app --reload
```

Приложение будет доступно на http://localhost:8000, документация API — http://localhost:8000/docs

## API

- `POST /learn` — загрузить знания текстом или по URL
- `POST /learn/file` — загрузить PDF/HTML/TXT файл
- `POST /query` — задать вопрос, получить ответ на основе базы знаний
- `GET /stats` — количество чанков в базе

Примеры запросов — в файле `api-collection.json`
