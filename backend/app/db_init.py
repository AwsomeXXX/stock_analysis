"""
db_init.py
- Create database tables (SQLModel metadata.create_all)
- Default DB URL: sqlite:///./data.db
"""
import os
from sqlmodel import SQLModel, create_engine

DEFAULT_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./data.db")

def create_database(db_url: str = DEFAULT_DB_URL):
    engine = create_engine(db_url, echo=False)
    SQLModel.metadata.create_all(engine)
    return engine

if __name__ == "__main__":
    engine = create_database()
    print("Database initialized/created for URL:", DEFAULT_DB_URL)
