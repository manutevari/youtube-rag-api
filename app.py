import os
import re
import subprocess
import streamlit as st
import tiktoken
from sqlalchemy import create_engine, text
from openai import OpenAI

# ============================================================
# 🔐 LOAD SECRETS (Streamlit Cloud + Local Compatible)
# ============================================================

OPENAI_API_KEY = None
DB_URL = None

# Streamlit Cloud secrets
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DB_URL = st.secrets["DB_URL"]

# Local .env fallback
else:
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DB_URL = os.getenv("DB_URL")

if not OPENAI_API_KEY or not DB_URL:
    st.error("Missing OPENAI_API_KEY or DB_URL")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DB_URL)

# ============================================================
# 🗄 INIT DATABASE
# ============================================================

def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS video_chunks (
                id SERIAL PRIMARY KEY,
                video_id TEXT,
                chunk_text TEXT,
                embedding VECTOR(1536),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))
        conn.commit()

init_db()

# ============================================================
# 🎬 UTILITIES
# ============================================================

def extract_video_id(url):
    pattern = r"(?:v=|youtu.be/)([^&]+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def download_audio(url, output="audio.mp3"):
    subprocess.run([
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "-o", output,
        url
    ], check=True)
    return output

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
    return transcript.text

def chunk_text(text, max_tokens=400):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    sentences = text.split(". ")

    chunks = []
    current = []
    token_count = 0

    for sentence in sentences:
        tokens = len(enc.encode(sentence))
        if token_count + tokens > max_tokens:
            chunks.append(" ".join(current))
            current = []
            token_count = 0
        current.append(sentence)
        token_count += tokens

    if current:
        chunks.append(" ".join(current))

    return chunks

def create_embeddings_batch(text_list):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in response.data]

# ============================================================
# 📦 INGEST VIDEO
# ============================================================

def ingest_video(youtube_url):

    video_id = extract_video_id(youtube_url)

    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT COUNT(*) FROM video_chunks WHERE video_id = :vid"),
            {"vid": video_id}
        ).scalar()

    if exists > 0:
        return video_id, "Video already ingested."

    audio_file = download_audio(youtube_url)
    transcript_text = transcribe_audio(audio_file)
    chunks = chunk_text(transcript_text)
    embeddings = create_embeddings_batch(chunks)

    with engine.connect() as conn:
        for chunk, emb in zip(chunks, embeddings):
            conn.execute(
                text("""
                    INSERT INTO video_chunks (video_id, chunk_text, embedding)
                    VALUES (:video_id, :chunk_text, :embedding)
                """),
                {
                    "video_id": video_id,
                    "chunk_text": chunk,
                    "embedding": emb
                }
            )
        conn.commit()

    # Clean up audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)

    return video_id, f"Ingested {len(chunks)} chunks."

# ============================================================
# 🔎 RAG QUESTION ANSWERING
# ============================================================

def answer_question(video_id, question, threshold=0.5):

    q_embedding = create_embeddings_batch([question])[0]

    query_sql = """
    SELECT chunk_text,
           embedding <-> CAST(:embedding AS vector) AS distance
    FROM video_chunks
    WHERE video_id = :video_id
    ORDER BY embedding <-> CAST(:embedding AS vector)
    LIMIT 5;
    """

    with engine.connect() as conn:
        results = conn.execute(
            text(query_sql),
            {"embedding": q_embedding, "video_id": video_id}
        ).fetchall()

    if not results:
        return "No relevant data found."

    if results[0][1] > threshold:
        return "Question not relevant to this video."

    context = "\n".join([row[0] for row in results])

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""
    )

    return response.output_text

# ============================================================
# 🎨 STREAMLIT UI
# ============================================================

st.title("🎥 YouTube RAG Assistant")

youtube_url = st.text_input("Enter YouTube URL")

if st.button("Ingest Video"):
    if youtube_url:
        with st.spinner("Processing video..."):
            try:
                vid, msg = ingest_video(youtube_url)
                st.session_state["video_id"] = vid
                st.success(msg)
            except Exception as e:
                st.error(f"Error: {e}")

if "video_id" in st.session_state:
    st.subheader("Ask Questions")
    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            try:
                answer = answer_question(st.session_state["video_id"], question)
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
