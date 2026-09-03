"""A few small tests to confirm the basics work.

They deliberately avoid calling the real embedding model or the real LLM, so
they run fast and need no API keys. Deeper tests come in a later phase.
"""

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.ingestion import split_into_chunks
from app.main import app
from app.models import QuestionRequest

client = TestClient(app)


def test_health_endpoint():
    """GET /health returns {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_question_request_model():
    """QuestionRequest accepts a plain question string."""
    model = QuestionRequest(question="What is RAG?")
    assert model.question == "What is RAG?"


def test_split_into_chunks_produces_multiple_chunks():
    """A long document is split into more than one chunk."""
    long_text = "This is a test sentence. " * 500
    chunks = split_into_chunks([Document(page_content=long_text)])
    assert len(chunks) > 1
