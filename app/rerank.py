"""Second-stage reranking for retrieval.

ChromaDB gives a fast but rough shortlist (`top_k` chunks). A CrossEncoder then
reads each (question, chunk) pair together and scores how well the chunk answers
the question - more accurate than the first-stage vector similarity. We keep
only the best `top_n` chunks and pass those to the LLM.

The model is a small open-source CrossEncoder (no API, no cost). It downloads
once on first use and then runs locally on CPU.
"""

from langchain_core.documents import Document

from app.config import settings

# Loaded once and reused, so we don't reload the model on every request.
_reranker = None


def _get_reranker():
    """Load the CrossEncoder the first time reranking is actually needed."""
    global _reranker
    if _reranker is None:
        # Imported here so this heavy library only loads when reranking runs.
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(question: str, chunks: list[Document], top_n: int) -> list[Document]:
    """Re-score `chunks` against `question` and return the best `top_n`."""
    # Nothing to reorder if we already have few enough chunks.
    if len(chunks) <= top_n:
        return chunks

    try:
        model = _get_reranker()
        pairs = [(question, chunk.page_content) for chunk in chunks]
        scores = model.predict(pairs)
    except Exception:
        # If the reranker can't load/run, keep the original ChromaDB order.
        return chunks[:top_n]

    ranked = sorted(zip(scores, chunks), key=lambda pair: float(pair[0]), reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]
