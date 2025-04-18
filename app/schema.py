from marshmallow import Schema, fields

class ChatSchema(Schema):
    """Схема для сообщений чата и передачи переменных для промптов."""
    chat_id = fields.String(
        required=True,
        description="Идентификатор чата",
        example="21a5c138-812b-4119-9258-bf3bda011f1d"
    )
    message = fields.String(
        required=True,
        description="Текст сообщения или ответ модели",
        example="Здравствуйте, я цифровой помощник..."
    )
    full_name = fields.String(
        required=False,
        description="Имя клиента для обращения",
        example="Иванов Иван Иванович"
    )
    account_number = fields.String(
        required=False,
        description="Лицевой счет",
        example="999888777"
    )
    debt_amount = fields.String(
        required=False,
        description="Сумма долга",
        example="10 000 руб."
    )
    address = fields.String(
        required=False,
        description="Адрес",
        example="Москва, ул. Ленина, д. 1"
    )
    company_name = fields.String(
        required=False,
        description="Название компании",
        example="МосЭнерго"
    )
    company_phone = fields.String(
        required=False,
        description="Телефон компании",
        example="8-800-123-45-67"
    )
    partial_payment_amount = fields.String(
        required=False,
        description="Сумма частичной оплаты",
        example="5 000 руб."
    )

class InputSchema(Schema):
    """Схема для удаления чата по chat_id."""
    chat_id = fields.String(
        required=True,
        description="Идентификатор чата для удаления",
        example="21a5c138-812b-4119-9258-bf3bda011f1d"
    )

class ErrorSchema(Schema):
    """Схема для ответов с ошибками."""
    message = fields.String(
        required=True,
        description="Текст сообщения об ошибке",
        example="chat not exists"
    )


# from marshmallow import Schema, fields

# class ChatSchema(Schema):
#     """Схема для сообщений чата."""
#     chat_id = fields.String(
#         description="Идентификатор чата", 
#         required=True, 
#         example='21a5c138-812b-4119-9258-bf3bda011f1d'
#     )
#     message = fields.String(
#         description="Текст сообщения или ответ модели", 
#         required=True, 
#         example='Здравствуйте, я цифровой помощник...'
#     )

# class InputSchema(Schema):
#     """Схема для входящих запросов."""
#     chat_id = fields.String(
#         description="Идентификатор чата для удаления", 
#         required=True, 
#         example='21a5c138-812b-4119-9258-bf3bda011f1d'
#     )

# class ErrorSchema(Schema):
#     """Схема для ответов с ошибками."""
#     message = fields.String(
#         description="Текст сообщения об ошибке", 
#         required=True, 
#         example='chat not exists'
#     )



# from marshmallow import Schema, fields

# class ChatSchema(Schema):
#     chat_id = fields.String(description="ИД чата", required=True, example='string')
#     message = fields.String(description="Ответ модели", required=True, example='string')

# class InputSchema(Schema):
#    number = fields.Int(description="Число", required=True, example=5)
#    power = fields.Int(description="Степень", required=True, example=2)

# class OutputSchema(Schema):
#    result = fields.Int(description="Результат", required=True, example=25)

# class ErrorSchema(Schema):
#    message = fields.String(description="Описание ошибки", required=True, example='string')