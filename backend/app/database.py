import os
from sqlmodel import SQLModel, create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db")
engine = create_engine(DATABASE_URL, echo=False)

def get_engine():
    return engine

def init_db():
    # 先导入 models，确保表模型注册到 metadata
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)