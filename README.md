# CognitiveVoiceAgent-API


# FastAPI Debt Collection Bot

Бот для автоматизации звонков по задолженностям с использованием FastAPI, LangChain и OpenAI.

## Структура проекта

```
.
├── app/
│   ├── __init__.py          # Инициализация приложения
│   ├── config.py            # Настройки приложения
│   ├── dependencies.py      # Зависимости для DI
│   ├── schemas.py           # Pydantic схемы
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py           # API эндпоинты
│   │   └── docs.py          # Документация
│   ├── utils/
│   │   ├── __init__.py
│   │   └── prompts.py       # Промпты для AI
│   ├── resources/           # Ресурсы (FAQ.docx)
│   └── vec_db/              # Векторная БД
├── main.py                  # Точка входа
├── requirements.txt         # Зависимости
├── .env.example            # Пример переменных окружения
└── README.md               # Документация
```

## Установка

1. Клонируйте репозиторий
2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # или
   venv\Scripts\activate  # Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Создайте файл `.env` на основе `.env.example` и заполните переменные:
   ```bash
   cp .env.example .env
   ```

5. Поместите файл `FAQ.docx` в папку `app/resources/`

## Запуск

### Режим разработки
```bash
python main.py
```

### Продакшен
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Документация

После запуска документация доступна по адресу:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Основные эндпоинты

### POST /api/get_new_chat_id
Создание нового чата
```json
{
  "full_name": "Иванов Иван Иванович",
  "company_name": "МосЭнерго"
}
```

### POST /api/chat
Отправка сообщения в чат
```json
{
  "chat_id": "21a5c138-812b-4119-9258-bf3bda011f1d",
  "message": "Да, это я",
  "full_name": "Иванов Иван Иванович",
  "account_number": "999888777",
  "debt_amount": "10 000 руб.",
  "address": "Москва, ул. Ленина, д. 1",
  "company_name": "МосЭнерго",
  "company_phone": "8-800-123-45-67",
  "partial_payment_amount": "5 000 руб."
}
```

### GET /api/get_all_chat_ids
Получение списка всех активных чатов

### POST /api/remove_chat_by_id
Удаление чата
```json
{
  "chat_id": "21a5c138-812b-4119-9258-bf3bda011f1d"
}
```

## Переменные окружения

- `OPENAI_API_KEY` - API ключ OpenAI
- `TELEGRAM_API_TOKEN` - Токен Telegram бота (опционально)
- `FULL_NAME` - ФИО должника по умолчанию
- `ACCOUNT_NUMBER` - Лицевой счет по умолчанию
- `DEBT_AMOUNT` - Сумма долга по умолчанию
- `ADDRESS` - Адрес по умолчанию
- `COMPANY_NAME` - Название компании по умолчанию
- `COMPANY_PHONE` - Телефон компании по умолчанию
- `PARTIAL_PAYMENT_AMOUNT` - Сумма частичной оплаты по умолчанию

## Особенности

- Использует FastAPI для высокой производительности
- Pydantic для валидации данных
- LangChain и LangGraph для управления диалогом
- FAISS для векторного поиска по FAQ
- Поддержка переменных в промптах
- Автоматическая документация API
- Статус диалога (in_progress/finished)

## Разработка

Для добавления новых эндпоинтов:
1. Добавьте схемы в `app/schemas.py`
2. Добавьте эндпоинт в `app/routes/api.py`
3. При необходимости обновите промпты в `app/utils/prompts.py`

## Миграция с Flask

Основные изменения при миграции с Flask:
- `@app.route` → `@router.post`/`@router.get`
- `request.get_json()` → Pydantic модели в параметрах
- `jsonify()` → возврат словарей или Pydantic моделей
- `current_app.config` → глобальные переменные с Depends
- Marshmallow схемы → Pydantic модели
- Синхронные функции → асинхронные (async/await)
