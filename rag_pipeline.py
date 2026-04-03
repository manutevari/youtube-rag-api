import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
engine = create_engine(os.getenv("DB_URL"))

def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def answer_question(video_id, question):
    return "Question is irrelevant"  # placeholder strict logic
