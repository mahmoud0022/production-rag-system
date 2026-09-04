"""Basic tests for the RAG API.

Everything external is faked in `conftest.py`, so these run fast and need no
Ollama, no embedding-model download, and no PostgreSQL.
"""

from langchain_core.documents import Document

from app.db import save_document
from app.ingestion import split_into_chunks
from app.models import QuestionRequest
from app.rag import retrieve_chunks


# --- health ---------------------------------------------------------------

def test_health_endpoint(client):
    """GET /health returns 200 and {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- chunking -----------------------------------------------------------------

def test_question_request_model():
    """QuestionRequest accepts a plain question string."""
    model = QuestionRequest(question="What is RAG?")
    assert model.question == "What is RAG?"


def test_split_into_chunks_produces_multiple_chunks():
    """A long document is split into more than one chunk."""
    long_text = "This is a test sentence. " * 500
    chunks = split_into_chunks([Document(page_content=long_text)])
    assert len(chunks) > 1


# --- upload -----------------------------------------------------------------

def test_upload_returns_filename_and_chunk_count(client, fake_store, monkeypatch):
    """POST /upload accepts a PDF and returns filename + chunks_added."""
    # Skip real PDF parsing - we're testing the endpoint, not pypdf.
    monkeypatch.setattr(
        "app.ingestion.load_pdf",
        lambda path: [Document(page_content="RAG means retrieval augmented generation. " * 60,
                               metadata={"page": 0})],
    )

    response = client.post(
        "/upload",
        files={"file": ("notes.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.pdf"
    assert body["chunks_added"] > 1
    # the chunks really landed in the (fake) vector store
    assert len(fake_store.docs) == body["chunks_added"]


# --- documents --------------------------------------------------------------

def test_documents_endpoint_lists_metadata(client):
    """GET /documents returns the rows saved in the database."""
    save_document("report.pdf", 7)

    response = client.get("/documents")

    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "report.pdf"
    assert docs[0]["chunks_added"] == 7
    assert "id" in docs[0]
    assert "uploaded_at" in docs[0]


# --- retrieval ------------------------------------------------------------------

def test_retrieval_returns_relevant_chunks(fake_store):
    """retrieve_chunks() puts the most relevant chunk first."""
    fake_store.add_documents([
        Document(page_content="The sky is blue during the day."),
        Document(page_content="Paris is the capital of France."),
        Document(page_content="Bananas are a yellow fruit."),
    ])

    chunks = retrieve_chunks("What colour is the sky?")

    assert len(chunks) >= 1
    assert "sky" in chunks[0].page_content.lower()


# --- ask ----------------------------------------------------------------------

def test_ask_returns_answer_and_sources(client, fake_store):
    """POST /ask returns an answer plus the source chunks it used."""
    fake_store.add_documents([
        Document(page_content="RAG combines retrieval with a language model.",
                 metadata={"page": 1}),
    ])

    response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This is a test answer."   # canned by the fake LLM
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["text"]
