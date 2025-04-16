from flask import Blueprint, current_app
from flask_swagger_ui import get_swaggerui_blueprint
import json

from app.schema import ChatSchema, InputSchema, OutputSchema, ErrorSchema

def create_tags(spec):
   """ Создаем теги.

   :param spec: объект APISpec для сохранения тегов
   """
   tags = [{'name': 'main', 'description': 'Основные запросы'}]

   for tag in tags:
       print(f"Добавляем тег: {tag['name']}")
       spec.tag(tag)



def load_docstrings(spec, app):
   """ Загружаем описание API.

   :param spec: объект APISpec, куда загружаем описание функций
   :param app: экземпляр Flask приложения, откуда берем описание функций
   """
   for fn_name in app.view_functions:
       if fn_name == 'static':
           continue
       print(f'Загружаем описание для функции: {fn_name}')
       view_fn = app.view_functions[fn_name]
       spec.path(view=view_fn)


from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin

def get_apispec(app):
   """ Формируем объект APISpec.

   :param app: объект Flask приложения
   """
   spec = APISpec(
       title="My App",
       version="1.0.0",
       openapi_version="3.0.3",
       plugins=[FlaskPlugin(), MarshmallowPlugin()],
   )

   spec.components.schema("Input", schema=InputSchema)
   spec.components.schema("ChatSchema", schema=ChatSchema)
   spec.components.schema("Error", schema=ErrorSchema)

   create_tags(spec)

   load_docstrings(spec, app)

   return spec

router = Blueprint(name='docs', import_name=__name__)



@router.route('/swagger')
def create_swagger_spec():
    return json.dumps(get_apispec(current_app).to_dict())

SWAGGER_URL = '/docs'
API_URL = '/swagger'

swagger_ui_blueprint = get_swaggerui_blueprint(
   SWAGGER_URL,
   API_URL,
   config={
       'app_name': 'My App'
   }
)