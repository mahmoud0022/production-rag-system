"""FastAPI application - the entry point.

Three endpoints:
  GET  /health   - is the API up?
  POST /upload   - send a PDF, it gets ingested into ChromaDB
  POST /ask      - ask a question, get an answer grounded in the uploaded PDFs

Run locally with:  uvicorn app.main:app --reload
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile

from app.ingestion import ingest_pdf
from app.models import AnswerResponse, QuestionRequest, UploadResponse
from app.rag import answer_question

app = FastAPI(title="Simple RAG System")

# Uploaded PDFs are saved here before ingestion.
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    """Quick check that the service is running."""
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile) -> UploadResponse:
    """Save an uploaded PDF to disk, then run the ingestion pipeline on it."""
    destination = UPLOAD_DIR / (file.filename or "upload.pdf")
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    chunks_added = ingest_pdf(str(destination))
    return UploadResponse(filename=destination.name, chunks_added=chunks_added)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:
    """Answer a question using the chunks already stored in ChromaDB."""
    return answer_question(request.question)
