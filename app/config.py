"""Application configuration.

Every setting (Ollama URL, model names, chunking parameters) is read once from
environment variables or a local `.env` file. Import `settings` wherever you
need a value - no other file should read the environment directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (local, served by Ollama - no API key, no cost) ---
    # Start Ollama with `ollama serve` and pull the model with
    # `ollama pull qwen2.5:3b` before asking questions.
    # Local default below; in Docker, docker-compose.yml sets
    # OLLAMA_BASE_URL=http://host.docker.internal:11434 (Ollama stays on the host).
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"

    # --- Embeddings ---
    # A small model that runs locally, so there is no extra API key and no cost.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- ChromaDB (stored as plain files on disk) ---
    chroma_dir: str = "./data/chroma"
    collection_name: str = "documents"

    # --- PostgreSQL (document metadata only; the RAG pipeline never uses it) ---
    # Local default below; in Docker, docker-compose.yml sets DATABASE_URL with
    # host `postgres` (the compose service name) instead of `localhost`.
    database_url: str = "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb"

    # --- How PDFs are split before embedding ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- How many chunks to retrieve for each question ---
    top_k: int = 8

    # --- Reranking (second stage: re-score the top_k chunks, keep the best) ---
    # Off by default; set USE_RERANKER=true to turn it on.
    use_reranker: bool = False
    # Small open-source CrossEncoder; downloaded locally on first use, no API.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many chunks survive reranking and are sent to the LLM.
    rerank_top_n: int = 4


# One shared instance used across the app.
settings = Settings()
