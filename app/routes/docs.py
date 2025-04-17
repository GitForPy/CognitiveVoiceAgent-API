from flask import Blueprint, current_app
from flask_swagger_ui import get_swaggerui_blueprint
import json
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin

from app.schema import ChatSchema, InputSchema, ErrorSchema  # Removed unused OutputSchema

def create_tags(spec):
    """Создаем теги."""
    tags = [{'name': 'main', 'description': 'Основные запросы'}]
    for tag in tags:
        print(f"Добавляем тег: {tag['name']}")
        spec.tag(tag)

def load_docstrings(spec, app):
    """Загружаем описание API."""
    for fn_name in app.view_functions:
        if fn_name == 'static':
            continue
        print(f'Загружаем описание для функции: {fn_name}')
        view_fn = app.view_functions[fn_name]
        spec.path(view=view_fn)

def get_apispec(app):
    """Формируем объект APISpec."""
    try:
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
    except Exception as e:
        print(f"Ошибка при формировании API спецификации: {e}")
        return None

router = Blueprint(name='docs', import_name=__name__)

@router.route('/swagger')
def create_swagger_spec():
    spec = get_apispec(current_app)
    if spec is None:
        return json.dumps({"error": "Failed to generate API specification"}), 500
    return json.dumps(spec.to_dict())

SWAGGER_URL = '/docs'
API_URL = '/swagger'

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'My App'
    }
)