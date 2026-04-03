import streamlit as st
from sqlalchemy import create_engine, text
from openai import OpenAI

st.set_page_config(page_title="RAG QA System", layout="wide")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
DB_URL = st.secrets["DB_URL"]

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DB_URL)

def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def answer_question(video_id, question):
    q_emb = get_embedding(question)

    sql = '''
        SELECT chunk_text, start_time, end_time,
               embedding <-> CAST(:emb AS vector) AS distance
        FROM video_chunks
        WHERE video_id = :vid
        ORDER BY embedding <-> CAST(:emb AS vector)
        LIMIT 5;
    '''

    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"emb": q_emb, "vid": video_id}).fetchall()

    if not rows:
        return "❌ Question is irrelevant"

    distances = [r[3] for r in rows]

    if distances[0] > 0.35:
        return "❌ Question is irrelevant"

    if len(distances) > 1 and distances[1] > 0.45:
        return "❌ Question is irrelevant"

    context = "\n\n".join(
        f"[{r[1]:.1f}s–{r[2]:.1f}s] {r[0]}"
        for r in rows
    )

    q_words = set(question.lower().split())
    c_words = set(context.lower().split())

    overlap = len(q_words & c_words) / max(1, len(q_words))

    if overlap < 0.2:
        return "❌ Question is irrelevant"

    confidence = (1 - distances[0]) * 0.7 + overlap * 0.3

    if confidence < 0.55:
        return "❌ Question is irrelevant"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer only if present, else say 'Question is irrelevant'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.0
    )

    answer = resp.choices[0].message.content.strip()

    if not answer or "irrelevant" in answer.lower():
        return "❌ Question is irrelevant"

    return answer

st.title("🎯 RAG QA System")

video_id = st.text_input("Video ID")
question = st.text_area("Ask a question")

if st.button("Ask"):
    if not video_id or not question:
        st.warning("Please fill all fields")
    else:
        with st.spinner("Thinking..."):
            result = answer_question(video_id, question)
            st.success(result)
