import os
import uuid
from pathlib import Path
from typing import Annotated, TypedDict, List, Dict

from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from prompts import DEBT_DISCUSSION_PROMPT, IDENTIFICATION_SYSTEM_PROMPT, ROUTER_ID_PROMPT, ROUTER_DEBT_PROMPT

# Конфигурация и пути
OPENAI_API_KEY = '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'
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


# Инициализация векторного хранилища
def init_vector_store() -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            folder_path=FAISS_INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    if not os.path.exists(DOCX_PATH):
        raise FileNotFoundError(f"Required document not found: {DOCX_PATH}")
    loader = UnstructuredWordDocumentLoader(DOCX_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

vector_store = init_vector_store()
LLM_CONVERSATIONAL = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_CONVERSATIONAL_MODEL, temperature=LLM_TEMPERATURE_CONV)
LLM_ROUTER = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_ROUTER_MODEL, temperature=LLM_TEMPERATURE_ROUTER)

# Определение типа состояния диалога
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    phase: str

def get_top_chunks_from_vector_db(query_text, vector_store, n=2):
    retriever = vector_store.as_retriever(search_kwargs={"k": n})
    docs = retriever.invoke(query_text)
    return " ".join([doc.page_content for doc in docs])

def run_llm_conversation_node(state: ConversationState, system_prompt: str) -> dict:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
    ])
    chain = prompt | LLM_CONVERSATIONAL | StrOutputParser()
    if "debt_discussion" in state["phase"].lower():
        last_user_text = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        retrieved_info = get_top_chunks_from_vector_db(last_user_text, vector_store, TOP_K_CHUNKS) if last_user_text else ""
        response = chain.invoke({
            "history": messages,
            "user_question": last_user_text,
            "retrieved_information": retrieved_info
        })
    else:
        response = chain.invoke({"history": messages})
    if not response or not response.strip():
        response = "Хорошо, в таком случае всего доброго, до свидания!"
    return {"messages": [AIMessage(content=response)]}

def node_identification(state: ConversationState) -> dict:
    messages = state["messages"]
    if not messages:
        initial_question = "Подскажите, пожалуйста, я сейчас разговариваю со Степанчук Аленой Сергеевной?"
        return {"messages": [AIMessage(content=initial_question)]}
    else:
        return run_llm_conversation_node(state, IDENTIFICATION_SYSTEM_PROMPT)

def node_debt_discussion(state: ConversationState) -> dict:
    return run_llm_conversation_node(state, DEBT_DISCUSSION_PROMPT)

def _get_llm_router_decision(state: ConversationState, router_prompt: str) -> str:
    if not state["messages"]:
        return "continue_identification" if "identification" in router_prompt.lower() else "continue_debt_discussion"
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(router_prompt),
        MessagesPlaceholder(variable_name="history_placeholder"),
    ])
    chain = prompt_template | LLM_ROUTER | StrOutputParser()
    decision = chain.invoke({"history_placeholder": state["messages"]}).strip().lower()
    valid_decisions_id = ['debt_discussion', 'end_conversation', 'continue_identification']
    valid_decisions_debt = ['end_conversation', 'continue_debt_discussion']
    if "identification" in router_prompt.lower():
        if decision not in valid_decisions_id:
            decision = "continue_identification"
    else:
        if decision not in valid_decisions_debt:
            decision = "continue_debt_discussion"
    return decision

def router_identification(state: ConversationState) -> str:
    decision = _get_llm_router_decision(state, ROUTER_ID_PROMPT)
    if decision not in ['debt_discussion', 'end_conversation', 'continue_identification']:
        decision = "continue_identification"
    return decision

def router_debt_discussion(state: ConversationState) -> str:
    decision = _get_llm_router_decision(state, ROUTER_DEBT_PROMPT)
    if decision not in ['end_conversation', 'continue_debt_discussion']:
        decision = "continue_debt_discussion"
    return decision

# Настройка графа диалога с использованием явных роутер-функций
workflow = StateGraph(ConversationState)
workflow.add_node("identification", node_identification)
workflow.add_node("debt_discussion", node_debt_discussion)
workflow.set_entry_point("identification")
workflow.add_conditional_edges(
    "identification",
    router_identification,
    {
        "continue_identification": "identification",
        "debt_discussion": "debt_discussion",
        "end_conversation": END
    }
)
workflow.add_conditional_edges(
    "debt_discussion",
    router_debt_discussion,
    {
        "continue_debt_discussion": "debt_discussion",
        "end_conversation": END
    }
)
agent = workflow.compile()