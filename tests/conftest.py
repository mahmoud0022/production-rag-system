"""Shared test fixtures.

Goal: a plain `pytest` run is fast, offline, and free - it never starts Ollama,
never downloads the embedding model, and never needs PostgreSQL. The `use_fakes`
fixture below swaps those parts out for every test.
"""

import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from sqlalchemy import create_engine

import app.db
import app.ingestion
import app.main
import app.rag
from app.db import Base


def _words(text: str) -> set[str]:
    """Lowercase words with punctuation stripped - used for fake 'similarity'."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class FakeVectorStore:
    """In-memory stand-in for the Chroma vector store.

    Keeps chunks in a list and 'retrieves' by counting words shared with the
    query. Enough to prove the wiring and a basic relevance contract without
    real embeddings or a running database.
    """

    def __init__(self) -> None:
        self.docs: list[Document] = []

    def add_documents(self, docs: list[Document]) -> None:
        self.docs.extend(docs)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        q = _words(query)
        ranked = sorted(
            self.docs,
            key=lambda d: len(q & _words(d.page_content)),
            reverse=True,
        )
        return ranked[:k]


@pytest.fixture
def fake_store() -> FakeVectorStore:
    """The single fake vector store used by a test (shared with the app)."""
    return FakeVectorStore()


@pytest.fixture(autouse=True)
def use_fakes(monkeypatch, tmp_path, fake_store):
    """Replace the slow/external pieces for every test:

    - get_vector_store -> FakeVectorStore   (no Chroma, no embedding-model download)
    - ask_llm          -> a canned string   (no Ollama / Qwen)
    - db.engine        -> a throwaway SQLite (no PostgreSQL)
    - UPLOAD_DIR       -> a temp folder      (don't write into the real data/ dir)
    """
    # Vector store - patched in both modules that look it up.
    monkeypatch.setattr("app.ingestion.get_vector_store", lambda: fake_store)
    monkeypatch.setattr("app.rag.get_vector_store", lambda: fake_store)

    # LLM call.
    monkeypatch.setattr("app.rag.ask_llm", lambda prompt: "This is a test answer.")

    # Database - a fresh SQLite file per test, same table schema as Postgres.
    test_engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr("app.db.engine", test_engine)
    Base.metadata.create_all(test_engine)

    # Uploaded files go to a temp directory.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr("app.main.UPLOAD_DIR", uploads)


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client (the `use_fakes` fixture is already applied)."""
    return TestClient(app.main.app)
