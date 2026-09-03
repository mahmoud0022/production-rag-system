"""Pydantic models describing the API's request and response bodies.

FastAPI uses these to validate incoming JSON, to serialise responses, and to
build the automatic documentation at `/docs`.
"""

from pydantic import BaseModel


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
