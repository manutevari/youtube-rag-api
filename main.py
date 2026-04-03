from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import answer_question

app = FastAPI()

class QueryRequest(BaseModel):
    video_id: str
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    return {"answer": answer_question(req.video_id, req.question)}
