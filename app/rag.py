"""Question answering: retrieve relevant chunks, then ask the LLM.

This is the "read" side of the system. `answer_question(question)` is the only
function the API needs to call.
"""

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from app.config import settings
from app.ingestion import get_vector_store
from app.models import AnswerResponse, Source
from app.rerank import rerank

# Kept simple and visible on purpose - this is the heart of "RAG".
PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""


def retrieve_chunks(question: str) -> list[Document]:
    """Return the `top_k` chunks from ChromaDB most similar to the question."""
    store = get_vector_store()
    return store.similarity_search(question, k=settings.top_k)


def build_prompt(question: str, chunks: list[Document]) -> str:
    """Join the retrieved chunks and the question into a single prompt string."""
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def ask_llm(prompt: str) -> str:
    """Send the prompt to the local Ollama model and return its plain-text answer."""
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)
    return llm.invoke(prompt).content


def answer_question(question: str) -> AnswerResponse:
    """Full RAG flow: retrieve -> (optional rerank) -> build prompt -> ask LLM -> response."""
    chunks = retrieve_chunks(question)                             # stage 1: ChromaDB, top_k

    # Stage 2 is optional. Off by default; enable with USE_RERANKER=true.
    if settings.use_reranker:
        chunks = rerank(question, chunks, settings.rerank_top_n)   # CrossEncoder, keep top_n

    prompt = build_prompt(question, chunks)
    answer = ask_llm(prompt)
    sources = [Source(text=c.page_content, page=c.metadata.get("page")) for c in chunks]
    return AnswerResponse(answer=answer, sources=sources)
