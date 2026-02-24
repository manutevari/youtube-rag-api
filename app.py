# ==========================================
# 🎓 STREAMLIT YOUTUBE RAG APP
# ==========================================

import streamlit as st
import os
import re
import pysrt
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BertTokenizer
from sqlalchemy import create_engine, text
from openai import OpenAI

nltk.download("punkt")

st.set_page_config(page_title="YouTube RAG", layout="wide")

st.title("🎥 YouTube RAG System")
st.write("Ask questions based on a YouTube video or uploaded transcript.")

# ==========================================
# 🔐 LOAD SECRETS
# ==========================================

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SUPABASE_DB_PASSWORD = st.secrets["SUPABASE_DB_PASSWORD"]

client = OpenAI(api_key=OPENAI_API_KEY)

DB_URL = (
    "postgresql://"
    "postgres.vvjsolwiuggknssusjfl:"
    f"{SUPABASE_DB_PASSWORD}"
    "@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres"
)

engine = create_engine(DB_URL, pool_pre_ping=True)

# ==========================================
# 🗄 INIT DATABASE
# ==========================================

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS video_chunks (
            id SERIAL PRIMARY KEY,
            video_id TEXT,
            chunk_text TEXT,
            start_time FLOAT,
            embedding VECTOR(1536)
        );
    """))
    conn.commit()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ==========================================
# 🧠 FUNCTIONS
# ==========================================

def embed_text(text_input):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_input
    )
    return response.data[0].embedding

def create_chunks(transcript, max_tokens=512):
    chunks = []
    current_text = []
    current_tokens = 0
    chunk_start = None

    for segment in transcript:
        sentence = segment["text"]
        start_time = segment["start"]

        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_count = len(tokens)

        if current_tokens + token_count > max_tokens:
            chunks.append({
                "text": " ".join(current_text),
                "start_time": chunk_start
            })
            current_text = []
            current_tokens = 0
            chunk_start = None

        if chunk_start is None:
            chunk_start = start_time

        current_text.append(sentence)
        current_tokens += token_count

    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start_time": chunk_start
        })

    return chunks

def store_chunks(video_id, transcript):

    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT COUNT(*) FROM video_chunks WHERE video_id = :vid"),
            {"vid": video_id}
        ).fetchone()

    if existing[0] > 0:
        return

    chunks = create_chunks(transcript)

    with engine.connect() as conn:
        for chunk in chunks:
            embedding = embed_text(chunk["text"])
            conn.execute(text("""
                INSERT INTO video_chunks
                (video_id, chunk_text, start_time, embedding)
                VALUES (:vid, :text, :start, :embedding)
            """), {
                "vid": video_id,
                "text": chunk["text"],
                "start": chunk["start_time"],
                "embedding": embedding
            })
        conn.commit()

def answer_question(question, threshold=0.5):

    query_embedding = embed_text(question)

    sql = """
    SELECT video_id, chunk_text, start_time,
           embedding <-> CAST(:embedding AS vector) AS distance
    FROM video_chunks
    ORDER BY embedding <-> CAST(:embedding AS vector)
    LIMIT 3;
    """

    with engine.connect() as conn:
        results = conn.execute(text(sql), {
            "embedding": query_embedding
        }).fetchall()

    if not results:
        return "No relevant data found."

    if results[0][3] > threshold:
        return "Question is irrelevant"

    video_id = results[0][0]
    start_time = int(results[0][2])
    link = f"https://www.youtube.com/watch?v={video_id}&t={start_time}"

    context = "\n".join([r[1] for r in results])

    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Answer ONLY using the context below.
If not found, say "Question is irrelevant".

Context:
{context}

Question:
{question}
"""
    )

    return f"""
Timestamp: {start_time}s  
Watch: {link}

Answer:
{response.output_text}
"""

# ==========================================
# 🎥 INPUT SECTION
# ==========================================

option = st.radio("Select Input Type", ["YouTube URL", "Upload SRT"])

video_id = None

if option == "YouTube URL":
    url = st.text_input("Enter YouTube URL")
    if st.button("Ingest Video") and url:
        video_id = re.search(r"(?:v=|youtu.be/)([^&]+)", url).group(1)
        transcript_obj = YouTubeTranscriptApi().fetch(video_id)
        transcript = [{"text": t.text, "start": t.start} for t in transcript_obj]
        store_chunks(video_id, transcript)
        st.success("Video ingested successfully!")

else:
    uploaded_file = st.file_uploader("Upload SRT file", type=["srt"])
    if uploaded_file and st.button("Ingest Transcript"):
        subs = pysrt.from_string(uploaded_file.read().decode("utf-8"))
        transcript = []
        for sub in subs:
            start_seconds = (
                sub.start.hours * 3600 +
                sub.start.minutes * 60 +
                sub.start.seconds
            )
            transcript.append({
                "text": sub.text.replace("\n", " "),
                "start": start_seconds
            })

        video_id = uploaded_file.name.replace(".srt", "")
        store_chunks(video_id, transcript)
        st.success("Transcript ingested successfully!")

# ==========================================
# 💬 QUESTION SECTION
# ==========================================

question = st.text_input("Ask a question")

if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        answer = answer_question(question)
        st.markdown(answer)
