"""
Database connection and session management.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 默认在 data 目录下生成 sqlite 数据库
db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
os.makedirs(db_dir, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(db_dir, 'ai_statement.db')}"

# connect_args={"check_same_thread": False} is needed for SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
