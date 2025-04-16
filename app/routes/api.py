import os
import logging
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal, Dict

import langgraph
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from app.utils.prompts import DEBT_DISCUSSION_PROMPT, IDENTIFICATION_SYSTEM_PROMPT, ROUTER_ID_PROMPT, ROUTER_DEBT_PROMPT


from flask import Blueprint, request, jsonify, current_app
import textwrap
import json
import uuid


# Конфигурация
OPENAI_API_KEY = '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'
TELEGRAM_API_TOKEN = "7498273994:AAHkkke_bzRjPS5yhE39dgYyBfv9PEVjVQg"

BASE_DIR = Path(__file__).parent
RESOURCES_DIR = BASE_DIR / 'resources'
VEC_DB_DIR = BASE_DIR / 'vec_db'
RESOURCES_DIR.mkdir(exist_ok=True)
VEC_DB_DIR.mkdir(exist_ok=True)

DOCX_PATH = str(RESOURCES_DIR / 'FAQ.docx')
FAISS_INDEX_PATH = str(VEC_DB_DIR / 'faiss_index')

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 2
MAX_TELEGRAM_MSG_SIZE = 4096

EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_CONVERSATIONAL_MODEL = "gpt-4o"
LLM_ROUTER_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE_CONV = 0.1
LLM_TEMPERATURE_ROUTER = 0.0



# Модели
LLM_CONVERSATIONAL = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.1)
LLM_ROUTER = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.0)

# Состояние диалога
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    phase: str

# Вспомогательные функции (без изменений)
def format_history_for_llm(messages: List[BaseMessage]) -> str:
    return "\n".join([f"{'Бот' if isinstance(m, AIMessage) else 'Пользователь'}: {m.content}" for m in messages])

def get_top_chunks_from_vector_db(query_text, vector_store, n=2):
    retriever = vector_store.as_retriever(search_kwargs={"k": n})
    docs = retriever.invoke(query_text)
    doc_contents = [doc.page_content for doc in docs]
    combined_text = " ".join(doc_contents)
    return combined_text

def chunk_text(text: str, max_size: int = MAX_TELEGRAM_MSG_SIZE) -> List[str]:
    return [text[i:i + max_size] for i in range(0, len(text), max_size)] if text else []

# Исправленный узел LLM
def run_llm_conversation_node(state: ConversationState, system_prompt: str) -> dict:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
    ])
    chain = prompt | LLM_CONVERSATIONAL | StrOutputParser()

    new_messages = []
    ask_for_input = bool(messages and isinstance(messages[-1], AIMessage))
    generate_bot_response = not ask_for_input or not messages or (messages and not isinstance(messages[-1], AIMessage))

    if generate_bot_response:
        try:
            if "debt_discussion" in state["phase"].lower():
                last_user_text = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                retrieved_info = get_top_chunks_from_vector_db(last_user_text, current_app.config['VECTOR_STORE'], TOP_K_CHUNKS) if last_user_text else ""
                response = chain.invoke({
                    "history": messages,
                    "user_question": last_user_text,
                    "retrieved_information": retrieved_info
                })
            else:
                response = chain.invoke({"history": messages})

            if response and response.strip():
                ai_message = AIMessage(content=response)
                new_messages.append(ai_message)
                # logger.info(f"Бот: {response}")
            else:
                response = "Хорошо, в таком случае всего доброго, до свидания!"
                new_messages.append(AIMessage(content=response))
                # logger.info(f"Бот (по умолчанию): {response}")
        except Exception as e:
            # logger.error(f"Ошибка в LLM: {e}")
            response = "Хорошо, в таком случае всего доброго, до свидания!"
            new_messages.append(AIMessage(content=response))
            # logger.info(f"Бот (ошибка): {response}")

    return {"messages": new_messages}

def node_identification(state: ConversationState) -> dict:
    messages = state["messages"]
    if not messages:
        initial_question = "Подскажите, пожалуйста, я сейчас разговариваю со Степанчук Аленой Сергеевной?"
        # logger.info(f"Бот: {initial_question}")
        return {"messages": [AIMessage(content=initial_question)]}
    else:
        return run_llm_conversation_node(state, IDENTIFICATION_SYSTEM_PROMPT)

def node_debt_discussion(state: ConversationState) -> dict:
    return run_llm_conversation_node(state, DEBT_DISCUSSION_PROMPT)

# Роутеры с выводом решения
def _get_llm_router_decision(state: ConversationState, router_prompt: str) -> str:
    if not state["messages"]:
        decision = "continue_identification" if "identification" in router_prompt.lower() else "continue_debt_discussion"
        # logger.info(f"Router decision (no messages): {decision}")
        return decision

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(router_prompt),
        MessagesPlaceholder(variable_name="history_placeholder"),
    ])
    chain = prompt_template | LLM_ROUTER | StrOutputParser()

    try:
        decision = chain.invoke({"history_placeholder": state["messages"]}).strip().lower()
        # logger.info(f"Router decision: {decision}")
    except Exception as e:
        # logger.error(f"Ошибка в роутере: {e}")
        decision = "continue_identification" if "identification" in router_prompt.lower() else "continue_debt_discussion"
        # logger.info(f"Router fallback decision: {decision}")

    valid_decisions_id = ['debt_discussion', 'end_conversation', 'continue_identification']
    valid_decisions_debt = ['end_conversation', 'continue_debt_discussion']
    is_id_router = "identification" in router_prompt.lower()

    if is_id_router and decision not in valid_decisions_id:
        # logger.warning(f"Некорректное решение роутера для identification: {decision}, возвращаем continue_identification")
        decision = "continue_identification"
    elif not is_id_router and decision not in valid_decisions_debt:
        # logger.warning(f"Некорректное решение роутера для debt_discussion: {decision}, возвращаем continue_debt_discussion")
        decision = "continue_debt_discussion"

    # logger.info(f"Final router decision: {decision}")
    return decision

# Настройка графа (заглушка, так как используем пошаговую логику)
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
    print('Получение списка id')
    current_sessions = current_app.config['SESSIONS_DICT']
    return jsonify(current_app.config['SESSIONS_DICT'])




@router.route('/get_new_chat_id', methods=['GET'])
def get_new_chat_id():
    """
   ---
   get:
     summary: Создает новый чат
     responses:
       '200':
         description: Возвращает новый chat_id и первое сообщение для пользователя
         content:
           application/json:
             schema: ChatSchema
       '400':
         description: Не передан обязательный параметр
         content:
           application/json:
             schema: ErrorSchema
     tags:
       - main
   """
    
    while True:
        new_id = str(uuid.uuid4())
        if new_id not in list(current_app.config['SESSIONS_DICT'].keys()):

            # current_app.config['SESSIONS_DICT'][new_id] = {'status': 'new', 'state': 'wait'}
            # return jsonify({'chat_id': new_id})

            greeting = textwrap.dedent("""\
                Здравствуйте. Меня зовут цифровой помощник, я представляю компанию ЭнергоСбыт. Звоню по вопросу задолженности по коммунальным услугам. В целях контроля качества разговор записывается. Подскажите, пожалуйста, я сейчас разговариваю со Степанчук Аленой Сергеевной?"""
            )
            # await send_response(message, greeting)

            # Инициализируем состояние с уже отправленным сообщением
            initial_state: ConversationState = {
                "messages": [AIMessage(content=greeting)],
                "phase": "identification"
            }
            current_app.config['SESSIONS_DICT'][new_id] = [initial_state]

            # return (json.dumps({"chat_id": new_id, "message": greeting})).decode('utf-8')

            return jsonify({"chat_id": new_id, "message": greeting})


@router.route('/chat', methods=['POST'])
def chat():
    """
---
post:
  summary: Ответ модели
  description: Отправляет сообщение модели и получает ответ. Убедитесь, что заголовок 'Content-Type' установлен на 'application/json'.
  parameters:
    - in: body
      name: chat
      schema: 
        $ref: ChatSchema
  responses:
    '200':
      description: Ответ модели на сообщение пользователя
      content:
        application/json:
          schema: 
            ChatSchema
    '400':
      description: Не передан обязательный параметр
      content:
        application/json:
          schema: 
            ErrorSchema
  tags:
    - main
"""

    data = request.get_json()
    print(type(data))
    print(jsonify(data))

    user_id = data['chat_id']
    user_text = data['message'].strip() if data['message'] else ""
    try:
        # current_state = user_states[user_id]
        current_state = current_app.config['SESSIONS_DICT'][user_id][0]
    except KeyError:
        return jsonify({"message": "chat not exists"})


    current_state["messages"].append(HumanMessage(content=user_text))
    phase = current_state["phase"]

    try:
        if phase == "identification":
            decision = _get_llm_router_decision(current_state, ROUTER_ID_PROMPT)
            if decision == "debt_discussion":
                current_state["phase"] = "debt_discussion"
                node_result = node_debt_discussion(current_state)
                current_state["messages"].extend(node_result["messages"])
                for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                    return jsonify({"chat_id": user_id, "message": msg.content})
                    # await send_response(message, msg.content)
            elif decision == "end_conversation":
                farewell = "Хорошо, в таком случае всего доброго, до свидания!"
                return jsonify({"chat_id": user_id, "message": farewell})
                # await send_response(message, farewell)
                # del user_states[user_id]
                logger.info(f"Диалог завершен для пользователя {user_id} с фразой: {farewell}")
            else:  # continue_identification
                node_result = node_identification(current_state)
                current_state["messages"].extend(node_result["messages"])
                for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                    return jsonify({"chat_id": user_id, "message": msg.content})
                    # await send_response(message, msg.content)

        elif phase == "debt_discussion":
            decision = _get_llm_router_decision(current_state, ROUTER_DEBT_PROMPT)
            if decision == "end_conversation":
                farewell = "Хорошо, в таком случае всего доброго, до свидания!"
                return jsonify({"chat_id": user_id, "message": farewell})
                # await send_response(message, farewell)
                # del user_states[user_id]
                # logger.info(f"Диалог завершен для пользователя {user_id} с фразой: {farewell}")
            else:  # continue_debt_discussion
                node_result = node_debt_discussion(current_state)
                current_state["messages"].extend(node_result["messages"])
                for msg in [m for m in node_result["messages"] if isinstance(m, AIMessage)]:
                    return jsonify({"chat_id": user_id, "message": msg.content})
                    # await send_response(message, msg.content)

        current_app.config['SESSIONS_DICT'][user_id] = current_state

    except Exception as e:
        # logger.exception(f"Error in message handling: {e}")
        farewell = "Хорошо, в таком случае всего доброго, до свидания!"
        return jsonify({"chat_id": user_id, "message": farewell})
        # await send_response(message, farewell)
        # del user_states[user_id]
        # logger.info(f"Диалог завершен для пользователя {user_id} с фразой при ошибке: {farewell}")

    

@router.route('/remove_chat_by_id', methods=['POST'])
def remove_chat_by_id():
    data:dict = request.get_json()
    chat_id = data['chat_id']
    del current_app.config['SESSIONS_DICT'][chat_id]
    return jsonify({'message': 'chat deleted'})
    