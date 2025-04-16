from marshmallow import Schema, fields

class ChatSchema(Schema):
    chat_id = fields.String(description="ИД чата", required=True, example='string')
    message = fields.String(description="Ответ модели", required=True, example='string')

class InputSchema(Schema):
   number = fields.Int(description="Число", required=True, example=5)
   power = fields.Int(description="Степень", required=True, example=2)

class OutputSchema(Schema):
   result = fields.Int(description="Результат", required=True, example=25)

class ErrorSchema(Schema):
   message = fields.String(description="Описание ошибки", required=True, example='string')