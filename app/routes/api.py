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

IDENTIFICATION_SYSTEM_PROMPT_FORMATTED = IDENTIFICATION_SYSTEM_PROMPT.format(
    FULL_NAME=settings.FULL_NAME,
    ACCOUNT_NUMBER=settings.ACCOUNT_NUMBER,
    DEBT_AMOUNT=settings.DEBT_AMOUNT,
    ADDRESS=settings.ADDRESS,
    COMPANY_NAME=settings.COMPANY_NAME,
    COMPANY_PHONE=settings.COMPANY_PHONE
)

DEBT_DISCUSSION_PROMPT_FORMATTED = DEBT_DISCUSSION_PROMPT.format(
    FULL_NAME=settings.FULL_NAME,
    ACCOUNT_NUMBER=settings.ACCOUNT_NUMBER,
    DEBT_AMOUNT=settings.DEBT_AMOUNT,
    ADDRESS=settings.ADDRESS,
    COMPANY_NAME=settings.COMPANY_NAME,
    COMPANY_PHONE=settings.COMPANY_PHONE,
    PARTIAL_PAYMENT_AMOUNT=settings.PARTIAL_PAYMENT_AMOUNT
)

ROUTER_ID_PROMPT_FORMATTED = ROUTER_ID_PROMPT.format(
    FULL_NAME=settings.FULL_NAME
)

ROUTER_DEBT_PROMPT_FORMATTED = ROUTER_DEBT_PROMPT.format(
    FULL_NAME=settings.FULL_NAME
)

class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    phase: str

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


def node_identification(state: ConversationState) -> dict:
    messages = state["messages"]
    if not messages:
        initial_question = f"Подскажите, пожалуйста, я сейчас разговариваю со {settings.FULL_NAME}?"
        return {"messages": [AIMessage(content=initial_question)]}
    else:
        return run_llm_conversation_node(state, IDENTIFICATION_SYSTEM_PROMPT_FORMATTED)

def node_debt_discussion(state: ConversationState) -> dict:
    return run_llm_conversation_node(state, DEBT_DISCUSSION_PROMPT_FORMATTED)

def _get_llm_router_decision(state: ConversationState, router_prompt: str) -> str:
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(router_prompt),
        MessagesPlaceholder(variable_name="history_placeholder"),
    ])
    chain = prompt_template | LLM_ROUTER | StrOutputParser()
    decision = chain.invoke({"history_placeholder": state["messages"]}).strip().lower()
    return decision

workflow = StateGraph(ConversationState)
workflow.add_node("identification", node_identification)
workflow.add_node("debt_discussion", node_debt_discussion)
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
      summary: Получение списка активных чатов
      description: Возвращает словарь всех активных чатов и их состояний
      tags:
        - main
      responses:
        200:
          description: Список активных чатов
          content:
            application/json:
              schema:
                type: object
                additionalProperties: true
    """
    return jsonify(current_app.config['SESSIONS_DICT'])

@router.route('/get_new_chat_id', methods=['GET'])
def get_new_chat_id():
    """
    ---
    get:
      summary: Создание нового чата
      description: Создает новый чат и возвращает его ID с приветственным сообщением
      tags:
        - main
      responses:
        200:
          description: Успешное создание чата
          content:
            application/json:
              schema: ChatSchema
    """
    while True:
        new_id = str(uuid.uuid4())
        if new_id not in list(current_app.config['SESSIONS_DICT'].keys()):
            greeting = textwrap.dedent(
                f"Здравствуйте. Меня зовут цифровой помощник, я представляю компанию {settings.COMPANY_NAME}. "
                f"Звоню по вопросу задолженности по коммунальным услугам. В целях контроля качества разговор записывается. "
                f"Подскажите, пожалуйста, я сейчас разговариваю со {settings.FULL_NAME}?"
            )
            initial_state: ConversationState = {
                "messages": [AIMessage(content=greeting)],
                "phase": "identification"
            }
            current_app.config['SESSIONS_DICT'][new_id] = [initial_state]
            return jsonify({"chat_id": new_id, "message": greeting})

@router.route('/chat', methods=['POST'])
def chat():
    """
    ---
    post:
      summary: Отправка сообщения в чат
      description: Отправляет сообщение в чат и получает ответ от бота
      tags:
        - main
      requestBody:
        required: true
        content:
          application/json:
            schema: ChatSchema
      responses:
        200:
          description: Ответ бота
          content:
            application/json:
              schema: ChatSchema
        400:
          description: Ошибка
          content:
            application/json:
              schema: ErrorSchema
    """
    data = request.get_json()
    user_id = data['chat_id']
    user_text = data['message'].strip() if data['message'] else ""
    try:
        current_state = current_app.config['SESSIONS_DICT'][user_id][0]
    except KeyError:
        return jsonify({"message": "chat not exists"})

    current_state["messages"].append(HumanMessage(content=user_text))
    phase = current_state["phase"]

    try:
        if phase == "identification":
            decision = _get_llm_router_decision(current_state, ROUTER_ID_PROMPT_FORMATTED)
            if decision == "debt_discussion":
                current_state["phase"] = "debt_discussion"
                node_result = node_debt_discussion(current_state)
                current_state["messages"].extend(node_result["messages"])
                for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                    return jsonify({"chat_id": user_id, "message": msg.content})
            elif decision == "end_conversation":
                farewell = "Хорошо, в таком случае всего доброго, до свидания!"
                return jsonify({"chat_id": user_id, "message": farewell})
            else:  # continue_identification
                node_result = node_identification(current_state)
                current_state["messages"].extend(node_result["messages"])
                for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                    return jsonify({"chat_id": user_id, "message": msg.content})

        elif phase == "debt_discussion":
            decision = _get_llm_router_decision(current_state, ROUTER_DEBT_PROMPT_FORMATTED)
            if decision == "end_conversation":
                farewell = "Хорошо, в таком случае всего доброго, до свидания!"
                return jsonify({"chat_id": user_id, "message": farewell})
            else:  # continue_debt_discussion
                node_result = node_debt_discussion(current_state)
                current_state["messages"].extend(node_result["messages"])
                for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                    return jsonify({"chat_id": user_id, "message": msg.content})

        current_app.config['SESSIONS_DICT'][user_id] = [current_state]

    except Exception:
        farewell = "Хорошо, в таком случае всего доброго, до свидания!"
        return jsonify({"chat_id": user_id, "message": farewell})

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

