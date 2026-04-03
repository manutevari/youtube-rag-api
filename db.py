from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv("DB_URL"))

def setup_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
