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


from flask import Flask

from app.routes import routers

def create_app():

    app = Flask(__name__)

    app.config['JSON_AS_ASCII'] = True
    app.config['SESSIONS_DICT'] = {'123': {'status': 'new', 'state': 'wait'}}

    
    for router in routers:
        app.register_blueprint(router)


    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Инициализация векторного хранилища
    def init_vector_store() -> FAISS:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                logger.info("Загрузка существующего индекса")
                # Явно указываем allow_dangerous_deserialization=True
                return FAISS.load_local(
                    folder_path=FAISS_INDEX_PATH,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
            logger.info("Создание нового индекса")
            if not os.path.exists(DOCX_PATH):
                raise FileNotFoundError(f"Required document not found: {DOCX_PATH}")
            loader = UnstructuredWordDocumentLoader(DOCX_PATH)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
            return vector_store
        except Exception as e:
            logger.error(f"Ошибка при инициализации векторного хранилища: {e}")
            raise

    
    app.config['VECTOR_STORE'] = init_vector_store()


    return app