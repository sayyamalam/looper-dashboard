import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

db_user = os.getenv("DB_USER", "postgres")
db_pass = os.getenv("DB_PASSWORD", "password")
db_host = os.getenv("DB_HOST", "localhost")
db_name = os.getenv("DB_NAME", "postgres")

url = f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
engine = create_engine(url, echo=False)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()
