from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models.db import Base


settings = get_settings()
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _add_missing_chunk_columns()


def _add_missing_chunk_columns() -> None:
    """Additive, idempotent upgrade for databases created before embeddings were stored."""
    existing = {column["name"] for column in inspect(engine).get_columns("chunks")}
    statements = []
    if "embedding" not in existing:
        statements.append("ALTER TABLE chunks ADD COLUMN embedding BLOB")
    if "embedding_model" not in existing:
        statements.append("ALTER TABLE chunks ADD COLUMN embedding_model VARCHAR(128)")
    if statements:
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))


def get_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

