from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
import config

load_dotenv()

CHROMA_PATH = "chromadb"
MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDINGS = HuggingFaceEmbeddings(model_name=MODEL_NAME)

print("Initializing LLM...")
LLM = ChatOpenAI(
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY,
    model=config.LLM_NAME,
    temperature=0.7,
)
print("LLM initialized.")

print("Initializing retriever...")
VECTOR_STORE = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=EMBEDDINGS
)
RETRIEVER = VECTOR_STORE.as_retriever(search_kwargs={"k": 4})
print("Retriever initialized.")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        sources: list of sources
    """
    question: str
    generation: str
    documents: List[Document]
    sources: List[str]

async def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = await RETRIEVER.ainvoke(question)
    sources = [doc.metadata['source'] for doc in documents]
    return {"documents": documents, "question": question, "sources": sources}

async def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    sources = state["sources"]

    prompt = PromptTemplate(
        template="""Вы — ассистент компании EORA. Ваша задача — отвечать на вопросы о выполненных проектах компании.
Используйте следующие фрагменты полученного контекста, чтобы ответить на вопрос.
В ответе должны быть упомянуты **все** проекты из списка источников.
В ответе должны быть ссылки на используемые файлы в формате HTML.
Например: "Мы делали бота для HR для <a href="https://example.com/magnit-bot">Магнита</a>, а ещё поиск по картинкам для <a href="https://example.com/kazan-express-search">KazanExpress</a>".
Если вы не знаете ответа, просто скажите, что не знаете.
Используйте максимум три предложения и давайте краткий ответ.

Вопрос: {question}
Контекст:
{context}

Источники:
{sources}

Ответ:""",
        input_variables=["question", "context", "sources"],
    )

    context = "\n".join([doc.page_content for doc in documents])
    sources_text = "\n".join([f"[{i+1}]: {source}" for i, source in enumerate(sources)])


    rag_chain = prompt | LLM | StrOutputParser()
    generation = await rag_chain.ainvoke({"context": context, "question": question, "sources": sources_text})
    return {"documents": documents, "question": question, "generation": generation, "sources": sources}

def get_rag_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)  

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

async def get_answer(query):
    """
    Gets an answer to a query using the RAG workflow.
    """
    print(f"Executing query: {query}")
    app = get_rag_workflow()
    response = await app.ainvoke({"question": query})
    return {
        "generation": response['generation'],
        "sources": response['sources']
    }