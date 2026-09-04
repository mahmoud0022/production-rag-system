"""Pydantic models describing the API's request and response bodies.

FastAPI uses these to validate incoming JSON, to serialise responses, and to
build the automatic documentation at `/docs`.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    """Returned by POST /upload after a PDF has been ingested."""

    filename: str
    chunks_added: int


class QuestionRequest(BaseModel):
    """Body of POST /ask - the user's question."""

    question: str


class Source(BaseModel):
    """One chunk of source text that was used to build the answer."""

    text: str
    page: int | None = None


class AnswerResponse(BaseModel):
    """Returned by POST /ask - the answer plus the chunks it was based on."""

    answer: str
    sources: list[Source]


class DocumentOut(BaseModel):
    """One row from the `documents` table (returned by GET /documents)."""

    # Allow building this model straight from a SQLAlchemy ORM object.
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    chunks_added: int
    uploaded_at: datetime
