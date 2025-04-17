
from marshmallow import Schema, fields

class ChatSchema(Schema):
    """Схема для сообщений чата."""
    chat_id = fields.String(
        description="Идентификатор чата", 
        required=True, 
        example='21a5c138-812b-4119-9258-bf3bda011f1d'
    )
    message = fields.String(
        description="Текст сообщения или ответ модели", 
        required=True, 
        example='Здравствуйте, я цифровой помощник...'
    )

class InputSchema(Schema):
    """Схема для входящих запросов."""
    chat_id = fields.String(
        description="Идентификатор чата для удаления", 
        required=True, 
        example='21a5c138-812b-4119-9258-bf3bda011f1d'
    )

class ErrorSchema(Schema):
    """Схема для ответов с ошибками."""
    message = fields.String(
        description="Текст сообщения об ошибке", 
        required=True, 
        example='chat not exists'
    )



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