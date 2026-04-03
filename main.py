# ============================================================
# 🔥 YOUTUBE RAG API (FASTAPI VERSION)
# ============================================================

import os
import re
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# 🔐 CONFIG
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DB_URL)

app = FastAPI(title="YouTube RAG API")

# ============================================================
# 📦 DATABASE SETUP
# ============================================================

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS video_chunks (
            id SERIAL PRIMARY KEY,
            video_id TEXT,
            chunk_text TEXT,
            embedding VECTOR(1536),
            start_time FLOAT,
            end_time FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """))
    conn.commit()

# ============================================================
# 📘 MODELS
# ============================================================

class IngestRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str

# ============================================================
# 🎬 UTILITIES
# ============================================================

def extract_video_id(url):
    pattern = r"(?:v=|youtu.be/)([^&]+)"
    match = re.search(pattern, url)
    return match.group(1)

def chunk_transcript(transcript, max_tokens=400):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    chunks = []
    current_text = []
    start_time = None
    token_count = 0

    for seg in transcript:
        text_seg = seg.text
        tokens = len(enc.encode(text_seg))

        if start_time is None:
            start_time = seg.start

        if token_count + tokens > max_tokens:
            chunks.append({
                "text": " ".join(current_text),
                "start": start_time,
                "end": seg.start
            })
            current_text = []
            start_time = seg.start
            token_count = 0

        current_text.append(text_seg)
        token_count += tokens

    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start": start_time,
            "end": transcript[-1].start
        })

    return chunks

def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]

# ============================================================
# 🚀 ENDPOINT: INGEST VIDEO
# ============================================================

@app.post("/ingest")
def ingest_video(request: IngestRequest):
    video_id = extract_video_id(request.url)
    transcript = YouTubeTranscriptApi().fetch(video_id)

    chunks = chunk_transcript(transcript)
    embeddings = create_embeddings([c["text"] for c in chunks])

    with engine.connect() as conn:
        for chunk, emb in zip(chunks, embeddings):
            conn.execute(text("""
                INSERT INTO video_chunks 
                (video_id, chunk_text, embedding, start_time, end_time)
                VALUES (:video_id, :chunk_text, :embedding, :start, :end)
            """), {
                "video_id": video_id,
                "chunk_text": chunk["text"],
                "embedding": emb,
                "start": chunk["start"],
                "end": chunk["end"]
            })
        conn.commit()

    return {"status": "Video ingested", "video_id": video_id}

# ============================================================
# 🔎 ENDPOINT: ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):
    q_embedding = create_embeddings([request.question])[0]

    query_sql = """
    SELECT video_id, chunk_text, start_time, end_time,
           embedding <-> CAST(:embedding AS
