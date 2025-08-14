from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from llm_logic import get_answer
from vector_store import create_vector_store_if_not_exists
from typing import List

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_vector_store_if_not_exists()
    yield

app = FastAPI(lifespan=lifespan)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    generation: str
    sources: List[str]

@app.post("/ask", response_model=Answer)
async def ask(question: Question):
    result = await get_answer(question.question)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)