# Production RAG System

A small retrieval-augmented generation (RAG) service: upload PDFs, ask questions,
and get answers grounded in the PDF content. It runs fully locally with no paid
APIs, and is also deployed to AWS.

---

## 1. What this project does

- A user uploads a PDF through the API.
- The text is extracted and split into overlapping chunks (~1000 characters).
- Each chunk is turned into an embedding (a vector) and stored in ChromaDB.
- The user asks a question.
- The most relevant chunks are retrieved from ChromaDB by similarity.
- Qwen 2.5 3B (served via Ollama) writes an answer using only those chunks.
- PostgreSQL stores metadata about each uploaded document (filename, chunk count, time).

---

## 2. Main technologies

- **FastAPI** – the web API (upload, ask, list documents, health).
- **ChromaDB** – local vector database that stores the chunk embeddings and text.
- **Sentence Transformers / MiniLM** (`all-MiniLM-L6-v2`) – turns chunks and questions into embeddings; runs locally.
- **Ollama** – runs the local language model and exposes it over HTTP.
- **Qwen 2.5 3B** – the open-weight LLM that writes the final answer.
- **PostgreSQL** – relational database for uploaded-document metadata.
- **Docker / Docker Compose** – runs FastAPI and PostgreSQL as containers.
- **Pytest** – automated tests for the pipeline and the API.
- **GitHub Actions CI/CD** – runs tests on every push/PR and deploys on success.
- **MLflow** – records evaluation runs (parameters + metrics) so configs can be compared.
- **AWS EC2** – the cloud server the project is deployed to.

---

## 3. Simple architecture

```
User
  |
FastAPI
  |
  +--> PDF ingestion --> chunks --> embeddings --> ChromaDB
  |
  +--> question --> retrieval --> optional reranking --> Qwen via Ollama
  |
  +--> PostgreSQL for document metadata
```

- FastAPI is the single entry point for everything.
- Ingestion is the "write" path: PDF in, embeddings into ChromaDB.
- Asking is the "read" path: retrieve chunks, optionally rerank, send them to Qwen.
- Reranking is off by default (it did not improve results — see section 8).
- PostgreSQL only stores document metadata; it is not used to answer questions.

---

## 4. Project structure

- `app/` – application code
  - `main.py` – FastAPI app and endpoints
  - `config.py` – all settings (models, chunking, DB URL, reranker flag)
  - `ingestion.py` – load PDF, chunk, embed, store in ChromaDB
  - `rag.py` – retrieve chunks, optional rerank, call the LLM
  - `rerank.py` – optional CrossEncoder reranker
  - `db.py` / `models.py` – PostgreSQL table + request/response schemas
- `tests/` – pytest suite (`test_basic.py`, fakes in `conftest.py`)
- `evaluation/` – evaluation script, question dataset, tuning notes
- `Dockerfile` – builds the FastAPI image (CPU-only PyTorch)
- `docker-compose.yml` – runs `api` + `postgres`
- `.github/workflows/ci.yml` – runs tests on GitHub
- `.github/workflows/cd.yml` – deploys to EC2 after CI passes
- `requirements.txt` – Python dependencies

---

## 5. How to run locally

Requires Python 3.11+ (tested on 3.13), Docker, and Ollama.

```bash
# 1. start PostgreSQL
docker compose up -d postgres

# 2. make sure Ollama is running (usually starts automatically after install)
ollama serve

# 3. pull the model (one-time, ~2 GB)
ollama pull qwen2.5:3b

# 4. install Python dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 5. run the API
uvicorn app.main:app --reload

# 6. open the interactive docs
#    http://localhost:8000/docs
```

Upload a PDF with `POST /upload` before asking questions.

---

## 6. Main API endpoints

- `GET /health` – returns `{"status": "ok"}` when the service is up.
- `POST /upload` – upload a PDF; it is chunked, embedded into ChromaDB, and recorded in PostgreSQL.
- `POST /ask` – send a question; returns an answer plus the source chunks used.
- `GET /documents` – list metadata for all uploaded PDFs, newest first.

---

## 7. Testing

- `pytest` runs **7 tests**: health, chunking, upload, documents, retrieval, and ask.
- The tests fake the external parts (Ollama, ChromaDB, and the PostgreSQL engine), so they are fast and need no running services.
- CI runs `pytest` automatically on every push and pull request to `main`.

```bash
pytest
```

---

## 8. Evaluation

- The evaluation runs **20 questions** based on `data/uploads/07_LiteratureReview.pdf`.
- Metrics:
  - **retrieval hit rate** – did the retrieved chunks contain the expected fact?
  - **answer accuracy** – did the generated answer contain the expected facts?
  - **average latency** – average time to answer one question.
- Best configuration:
  - `chunk_size = 1000`
  - `chunk_overlap = 150`
  - `top_k = 8`
  - reranker disabled by default
- Best measured result:
  - retrieval hit rate: **85%**
  - answer accuracy: **100%**
  - average latency: **~6.5–6.9 s** locally

Reranking (a CrossEncoder second pass) was tested. It did not improve retrieval
and slightly lowered answer accuracy while adding latency, so it stays optional
and off by default (set `USE_RERANKER=true` to try it).

---

## 9. MLflow

- Every evaluation run can be logged as one MLflow run.
- It stores the configuration (parameters) and the results (metrics).
- This makes it easy to compare different settings over time.

```bash
python -m evaluation.evaluate
mlflow ui --host 127.0.0.1 --port 5000 --workers 1
```

Open http://127.0.0.1:5000 and look at the `rag-evaluation` experiment.
Local MLflow data (`mlruns/`, `mlflow.db`) is not committed to Git.

---

## 10. Docker

- FastAPI and PostgreSQL run in containers via Docker Compose.
- Ollama runs **outside** Docker, on the host; the container reaches it at `host.docker.internal`.
- ChromaDB files are persisted in `./data` (bind mount).
- PostgreSQL data is persisted in a Docker named volume.
- The image installs **CPU-only PyTorch** so it does not pull large CUDA/NVIDIA packages on CPU-only machines.

```bash
docker compose up --build -d
docker compose ps
docker compose down
```

---

## 11. AWS deployment

- The project is deployed on an **AWS EC2** instance (Ubuntu).
- Docker runs **FastAPI + PostgreSQL** on the instance.
- **Ollama and Qwen run directly on the EC2 host**, not in Docker.
- ChromaDB persists under `data/` on the instance.
- The API is exposed on **port 8000**.
- Qwen runs on CPU there, so answers are slower than on a local machine with better hardware.

The public IP can change, so it is not written here.

---

## 12. CI/CD

**CI** (`.github/workflows/ci.yml`)

```
push / pull request --> GitHub Actions --> install deps --> pytest
```

**CD** (`.github/workflows/cd.yml`)

```
CI succeeds --> GitHub Actions --> AWS SSM --> EC2 --> git pull --> Docker rebuild/restart
```

- Deployment uses **AWS Systems Manager (SSM)** to run commands on the instance.
- No direct SSH from GitHub is needed.
- AWS credentials and the instance ID are stored as **GitHub Secrets**.

---

## 13. Important persistence

- **Uploaded PDFs + ChromaDB** – `./data` (bind mount; kept between runs).
- **PostgreSQL** – Docker named volume (`postgres_data`).
- **MLflow experiment data** – local only (`mlruns/`, `mlflow.db`); ignored by Git.

---

## 14. Current final setup

- Local RAG pipeline works.
- Docker (FastAPI + PostgreSQL) works.
- PostgreSQL document metadata works.
- Tests pass (7/7).
- CI works.
- Evaluation works.
- MLflow tracking works.
- AWS EC2 deployment works.
- CD (auto-deploy after CI) works.

---

## 15. Future improvements

- HTTPS and a domain name.
- A stronger cloud instance or a GPU for faster Qwen inference.
- Managed PostgreSQL (e.g. AWS RDS).
- Better monitoring.
- Authentication for the API.
