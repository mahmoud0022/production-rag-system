"""PostgreSQL access for document metadata.

Deliberately tiny: one engine, one table, three helper functions. The RAG
pipeline (ChromaDB, embeddings, Ollama) does not use this file at all - it only
records "which PDFs were uploaded and how many chunks each produced".
"""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import settings

# One shared engine for the whole app. `pool_pre_ping` quietly replaces
# connections that the database dropped while idle.
engine = create_engine(settings.database_url, pool_pre_ping=True)


class Base(DeclarativeBase):
    """Base class all ORM models inherit from."""


class Document(Base):
    """One row per successfully ingested PDF."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    chunks_added: Mapped[int] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def create_tables() -> None:
    """Create the `documents` table if it does not exist yet. Called on startup."""
    Base.metadata.create_all(engine)


def save_document(filename: str, chunks_added: int) -> None:
    """Insert one row recording a finished upload."""
    with Session(engine) as session:
        session.add(Document(filename=filename, chunks_added=chunks_added))
        session.commit()


def list_documents() -> list[Document]:
    """Return every document row, newest first."""
    with Session(engine) as session:
        rows = session.scalars(select(Document).order_by(Document.uploaded_at.desc()))
        return list(rows)
