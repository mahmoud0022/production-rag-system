"""FastAPI application - the entry point.

Endpoints:
  GET  /health     - is the API up?
  POST /upload     - send a PDF, it gets ingested into ChromaDB + recorded in PostgreSQL
  POST /ask        - ask a question, get an answer grounded in the uploaded PDFs
  GET  /documents  - list the PDFs that have been uploaded (metadata from PostgreSQL)

Run locally with:  uvicorn app.main:app --reload
"""

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile

from app.db import create_tables, list_documents, save_document
from app.ingestion import ingest_pdf
from app.models import AnswerResponse, DocumentOut, QuestionRequest, UploadResponse
from app.rag import answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: make sure the `documents` table exists."""
    create_tables()
    yield


app = FastAPI(title="Simple RAG System", lifespan=lifespan)

# Uploaded PDFs are saved here before ingestion.
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    """Quick check that the service is running."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile) -> UploadResponse:
    """Save the PDF, ingest it into ChromaDB, then record its metadata in PostgreSQL."""
    destination = UPLOAD_DIR / (file.filename or "upload.pdf")
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    chunks_added = ingest_pdf(str(destination))          # unchanged RAG pipeline
    save_document(destination.name, chunks_added)        # new: metadata row in Postgres
    return UploadResponse(filename=destination.name, chunks_added=chunks_added)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:
    """Answer a question using the chunks already stored in ChromaDB."""
    return answer_question(request.question)


@app.get("/documents", response_model=list[DocumentOut])
def documents() -> list[DocumentOut]:
    """List metadata for every uploaded PDF, newest first."""
    return list_documents()
