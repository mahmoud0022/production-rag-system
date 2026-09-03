"""Ingestion pipeline: PDF file -> text -> chunks -> embeddings -> ChromaDB.

This is the "write" side of the system. Call `ingest_pdf(path)` once for each
uploaded file. Reading the stored chunks back happens in `rag.py`.
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def load_pdf(path: str) -> list[Document]:
    """Read a PDF from disk and return one Document per page (keeps page numbers)."""
    return PyPDFLoader(path).load()


def split_into_chunks(pages: list[Document]) -> list[Document]:
    """Split page-sized text into smaller overlapping chunks.

    Smaller chunks make retrieval more precise. The overlap keeps a sentence
    from being lost when it falls exactly on a chunk boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(pages)


def get_vector_store() -> Chroma:
    """Open (or create) the on-disk ChromaDB collection.

    The same collection is used by ingestion (to add chunks) and by `rag.py`
    (to search them), so this helper lives here and is imported there.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )


def ingest_pdf(path: str) -> int:
    """Run the whole pipeline for one PDF. Returns how many chunks were stored.

    Steps: load pages -> split into chunks -> embed each chunk and write it to
    ChromaDB.
    """
    pages = load_pdf(path)
    chunks = split_into_chunks(pages)
    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)
