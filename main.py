from fastapi import FastAPI
from pydantic import BaseModel
from llm_logic import get_answer
from typing import List

app = FastAPI()

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
    uvicorn.run(app)