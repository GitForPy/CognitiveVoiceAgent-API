import uuid
import textwrap
from pathlib import Path
from typing import Annotated, TypedDict, List

from flask import Blueprint, request, jsonify, current_app

from settings import Settings
from app.utils.prompts import (
    DEBT_DISCUSSION_PROMPT,
    IDENTIFICATION_SYSTEM_PROMPT,
    ROUTER_ID_PROMPT,
    ROUTER_DEBT_PROMPT,
)

from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI

# --- Конфигурация и настройки ---
BASE_DIR = Path(__file__).parent
RESOURCES_DIR = BASE_DIR / 'resources'
VEC_DB_DIR = BASE_DIR / 'vec_db'
RESOURCES_DIR.mkdir(exist_ok=True)
VEC_DB_DIR.mkdir(exist_ok=True)

settings = Settings()
OPENAI_API_KEY = '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'
LLM_CONVERSATIONAL_MODEL = "gpt-4o"
LLM_ROUTER_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE_CONV = 0.1
LLM_TEMPERATURE_ROUTER = 0.0

LLM_CONVERSATIONAL = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=LLM_CONVERSATIONAL_MODEL,
    temperature=LLM_TEMPERATURE_CONV
)
LLM_ROUTER = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=LLM_ROUTER_MODEL,
    temperature=LLM_TEMPERATURE_ROUTER
)

class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    phase: str
    prompt_vars: dict

def run_llm_conversation_node(state: ConversationState, system_prompt: str) -> dict:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
    ])
    chain = prompt | LLM_CONVERSATIONAL | StrOutputParser()

    if "debt_discussion" in state["phase"].lower():
        last_user_text = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        response = chain.invoke({
            "history": messages,
            "user_question": last_user_text,
            "retrieved_information": ""
        })
    else:
        response = chain.invoke({"history": messages})

    response = (response or "").strip()
    if not response:
        response = "Извините, произошла техническая ошибка. Пожалуйста, повторите ваш вопрос."
    return {"messages": [AIMessage(content=response)]}

def _get_llm_router_decision(state: ConversationState, router_prompt: str) -> str:
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(router_prompt),
        MessagesPlaceholder(variable_name="history_placeholder"),
    ])
    chain = prompt_template | LLM_ROUTER | StrOutputParser()
    decision = chain.invoke({"history_placeholder": state["messages"]}).strip().lower()
    return decision

workflow = StateGraph(ConversationState)
workflow.add_node("identification", lambda state: run_llm_conversation_node(state, IDENTIFICATION_SYSTEM_PROMPT.format(**state.get("prompt_vars", {}))))
workflow.add_node("debt_discussion", lambda state: run_llm_conversation_node(state, DEBT_DISCUSSION_PROMPT.format(**state.get("prompt_vars", {}))))
workflow.set_entry_point("identification")
workflow.add_conditional_edges("identification", lambda state: "identification", {"identification": "identification", "debt_discussion": "debt_discussion", "__end__": END})
workflow.add_conditional_edges("debt_discussion", lambda state: "debt_discussion", {"debt_discussion": "debt_discussion", "__end__": END})
agent = workflow.compile()

router = Blueprint(name='api', import_name=__name__, url_prefix='/api')


@router.route('/get_all_chat_ids', methods=['GET'])
def get_all_chat_ids():
    """
    ---
    get:
      summary: Получение списка chat_id
      description: Возвращает список всех активных chat_id
      tags:
        - main
      responses:
        200:
          description: Список chat_id
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
    """
    sessions = current_app.config.get('SESSIONS_DICT', {})
    return jsonify(list(sessions.keys()))

@router.route('/get_new_chat_id', methods=['POST'])
def get_new_chat_id():
    """
    ---
    post:
      summary: Создание нового чата
      description: Создает новый чат и возвращает его ID с приветственным сообщением. Можно передать переменные для приветствия.
      tags:
        - main
      requestBody:
        required: false
        content:
          application/json:
            schema:
              type: object
              properties:
                full_name:
                  type: string
                  description: Имя клиента для приветствия
                  example: "Иванов Иван Иванович"
                company_name:
                  type: string
                  description: Название компании
                  example: "МосЭнерго"
      responses:
        200:
          description: Успешное создание чата
          content:
            application/json:
              schema: ChatSchema
    """
    data = request.get_json(silent=True) or {}
    full_name = data.get("full_name", settings.FULL_NAME)
    company_name = data.get("company_name", settings.COMPANY_NAME)

    while True:
        new_id = str(uuid.uuid4())
        if new_id not in list(current_app.config['SESSIONS_DICT'].keys()):
            greeting = textwrap.dedent(
                f"Здравствуйте. Меня зовут цифровой помощник, я представляю компанию {company_name}. "
                f"Звоню по вопросу задолженности по коммунальным услугам. В целях контроля качества разговор записывается. "
                f"Подскажите, пожалуйста, я сейчас разговариваю c {full_name}?"
            )
            initial_state: ConversationState = {
                "messages": [AIMessage(content=greeting)],
                "phase": "identification",
                "prompt_vars": {
                    "FULL_NAME": full_name,
                    "COMPANY_NAME": company_name
                }
            }
            current_app.config['SESSIONS_DICT'][new_id] = [initial_state]
            return jsonify({"chat_id": new_id, "message": greeting})

@router.route('/chat', methods=['POST'])
def chat():
    """
    ---
    post:
      summary: Отправка сообщения в чат
      description: |
        Отправляет сообщение в чат и получает ответ от бота. Можно передавать любые переменные для подстановки в промпты.
        В ответе всегда возвращается статус диалога:
        - "in_progress" — диалог продолжается
        - "finished" — диалог завершён (бот попрощался)
      tags:
        - main
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                chat_id:
                  type: string
                  description: Идентификатор чата
                  example: "21a5c138-812b-4119-9258-bf3bda011f1d"
                message:
                  type: string
                  description: Сообщение пользователя
                  example: "Да, я Иван Иванович"
                full_name:
                  type: string
                  description: Имя клиента для обращения
                  example: "Иванов Иван Иванович"
                account_number:
                  type: string
                  description: Лицевой счет
                  example: "999888777"
                debt_amount:
                  type: string
                  description: Сумма долга
                  example: "10 000 руб."
                address:
                  type: string
                  description: Адрес
                  example: "Москва, ул. Ленина, д. 1"
                company_name:
                  type: string
                  description: Название компании
                  example: "МосЭнерго"
                company_phone:
                  type: string
                  description: Телефон компании
                  example: "8-800-123-45-67"
                partial_payment_amount:
                  type: string
                  description: Сумма частичной оплаты
                  example: "5 000 руб."
      responses:
        200:
          description: Ответ бота
          content:
            application/json:
              schema:
                type: object
                properties:
                  chat_id:
                    type: string
                  message:
                    type: string
                  status:
                    type: string
                    description: Статус диалога ("in_progress" или "finished")
        400:
          description: Ошибка
          content:
            application/json:
              schema: ErrorSchema
    """
    data = request.get_json()
    user_id = data['chat_id']
    user_text = data['message'].strip() if data['message'] else ""

    prompt_vars = {
        "FULL_NAME": data.get("full_name", settings.FULL_NAME),
        "ACCOUNT_NUMBER": data.get("account_number", settings.ACCOUNT_NUMBER),
        "DEBT_AMOUNT": data.get("debt_amount", settings.DEBT_AMOUNT),
        "ADDRESS": data.get("address", settings.ADDRESS),
        "COMPANY_NAME": data.get("company_name", settings.COMPANY_NAME),
        "COMPANY_PHONE": data.get("company_phone", settings.COMPANY_PHONE),
        "PARTIAL_PAYMENT_AMOUNT": data.get("partial_payment_amount", settings.PARTIAL_PAYMENT_AMOUNT),
    }

    try:
        current_state = current_app.config['SESSIONS_DICT'][user_id][0]
    except KeyError:
        return jsonify({"message": "chat not exists"}), 400

    current_state["messages"].append(HumanMessage(content=user_text))
    current_state["prompt_vars"] = prompt_vars
    phase = current_state["phase"]

    if phase == "identification":
        decision = _get_llm_router_decision(current_state, ROUTER_ID_PROMPT.format(FULL_NAME=prompt_vars["FULL_NAME"]))
        if decision == "debt_discussion":
            current_state["phase"] = "debt_discussion"
            node_result = run_llm_conversation_node(current_state, DEBT_DISCUSSION_PROMPT.format(**prompt_vars))
            current_state["messages"].extend(node_result["messages"])
            for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                return jsonify({"chat_id": user_id, "message": msg.content, "status": "in_progress"})
        elif decision == "end_conversation":
            farewell = "Хорошо, в таком случае всего доброго, до свидания!"
            return jsonify({"chat_id": user_id, "message": farewell, "status": "finished"})
        else:  # continue_identification
            node_result = run_llm_conversation_node(current_state, IDENTIFICATION_SYSTEM_PROMPT.format(**prompt_vars))
            current_state["messages"].extend(node_result["messages"])
            for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                return jsonify({"chat_id": user_id, "message": msg.content, "status": "in_progress"})

    elif phase == "debt_discussion":
        decision = _get_llm_router_decision(current_state, ROUTER_DEBT_PROMPT.format(FULL_NAME=prompt_vars["FULL_NAME"]))
        if decision == "end_conversation":
            farewell = "Хорошо, в таком случае всего доброго, до свидания!"
            return jsonify({"chat_id": user_id, "message": farewell, "status": "finished"})
        else:  # continue_debt_discussion
            node_result = run_llm_conversation_node(current_state, DEBT_DISCUSSION_PROMPT.format(**prompt_vars))
            current_state["messages"].extend(node_result["messages"])
            for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                return jsonify({"chat_id": user_id, "message": msg.content, "status": "in_progress"})

    current_app.config['SESSIONS_DICT'][user_id] = [current_state]

@router.route('/remove_chat_by_id', methods=['POST'])
def remove_chat_by_id():
    """
    ---
    post:
      summary: Удаление чата
      description: Удаляет чат по его идентификатору
      tags:
        - main
      requestBody:
        required: true
        content:
          application/json:
            schema: InputSchema
      responses:
        200:
          description: Чат успешно удален
          content:
            application/json:
              schema: ErrorSchema
    """
    data = request.get_json()
    chat_id = data['chat_id']
    del current_app.config['SESSIONS_DICT'][chat_id]
    return jsonify({'message': 'chat deleted'})