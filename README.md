# production-rag-system

A small, readable **Retrieval-Augmented Generation (RAG)** service.

Upload a PDF, then ask questions about it. The API finds the most relevant
passages and asks a local open-weight LLM (Qwen2.5 3B, served by
[Ollama](https://ollama.com)) to answer using only those passages. No API key,
no cost - everything runs on your machine.

This is **phase 1**: the smallest version that actually works. It is built to be
easy to read and easy to explain, and to grow later (see
[Roadmap](#roadmap-later-phases)).

---

## What is RAG? (the pipeline, step by step)

RAG = give the language model the right context *before* it answers, instead of
hoping it already memorised the answer.

```
                        INGESTION (happens once per PDF)
  ┌────────┐   ┌───────────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
  │  PDF   │ → │ extract text  │ → │  chunk   │ → │  embed     │ → │ ChromaDB │
  │ upload │   │ (per page)    │   │ the text │   │ (vectors)  │   │ (store)  │
  └────────┘   └───────────────┘   └──────────┘   └────────────┘   └──────────┘

                        ASKING (happens for every question)
  ┌──────────┐   ┌────────────────────┐   ┌──────────────────┐   ┌───────────┐
  │ question │ → │ embed the question │ → │ ChromaDB returns │ → │ build a   │
  │          │   │ + similarity search│   │ top-k chunks     │   │ prompt    │
  └──────────┘   └────────────────────┘   └──────────────────┘   └─────┬─────┘
                                                                       ▼
                                        ┌──────────┐          ┌─────────────────┐
                                        │  answer  │  ◄────── │ LLM (Ollama)    │
                                        │ + sources│          │ answers using   │
                                        └──────────┘          │ only the chunks │
                                                              └─────────────────┘
```

1. **Upload a PDF** – `POST /upload`.
2. **Extract text** – read the PDF one page at a time (`pypdf`).
3. **Chunk** – cut each page into ~1000-character overlapping pieces. Small
   pieces are easier to match to a specific question.
4. **Embed** – turn each chunk into a vector (a list of numbers) with a small
   embedding model that runs locally. Similar meaning → similar vector.
5. **Store** – save the vectors and their text in **ChromaDB**, a local vector
   database (just files on disk).
6. **Ask a question** – `POST /ask`.
7. **Retrieve** – embed the question the same way and ask ChromaDB for the
   `top_k` most similar chunks.
8. **Prompt + LLM** – paste those chunks and the question into a prompt template
   and send it to the local LLM (Qwen2.5 3B via Ollama).
9. **Respond** – return the answer plus the chunks it was based on, as JSON.

---

## Folder structure

```
production-rag-system/
├── app/
│   ├── __init__.py       marks "app" as a Python package
│   ├── main.py           FastAPI app + endpoints: /health, /upload, /ask
│   ├── config.py         all settings & model names, loaded from .env
│   ├── ingestion.py      PDF -> text -> chunks -> embeddings -> ChromaDB
│   ├── rag.py            retrieve chunks -> build prompt -> call the LLM
│   └── models.py         Pydantic request/response models
├── data/                 uploaded PDFs + the ChromaDB files live here (git-ignored)
├── tests/
│   └── test_basic.py     a few fast tests (no API key needed)
├── conftest.py           lets tests import the "app" package
├── .env.example          template for your .env
├── requirements.txt      dependencies
├── .gitignore
└── README.md
```

### What each file does

| File | Responsibility |
|------|----------------|
| `app/main.py` | Defines the FastAPI app and the three HTTP endpoints. `/upload` saves the file and calls `ingest_pdf`; `/ask` calls `answer_question`. Thin - no logic of its own. |
| `app/config.py` | One `Settings` class (from `pydantic-settings`) holding the Ollama URL, model names, chunk size, `top_k`, etc. Exposes a single `settings` object. |
| `app/ingestion.py` | The "write" path. `load_pdf`, `split_into_chunks`, `get_vector_store`, and `ingest_pdf` which runs them in order. |
| `app/rag.py` | The "read" path. `retrieve_chunks`, `build_prompt`, `ask_llm`, and `answer_question` which ties them together. Contains the prompt template. |
| `app/models.py` | `UploadResponse`, `QuestionRequest`, `Source`, `AnswerResponse` - the JSON shapes, used by FastAPI for validation and `/docs`. |
| `tests/test_basic.py` | Checks `/health`, the `QuestionRequest` model, and that chunking splits a long text. Runs without keys or network. |
| `conftest.py` | Empty except for a docstring; its presence makes `import app...` work in tests. |

---

## Setup & run

Requires Python 3.11+ (tested on 3.13) and [Ollama](https://ollama.com/download).

```bash
# 1. get the local LLM (one-time; ~2 GB download)
ollama pull qwen2.5:3b
ollama serve                      # keep running; on Windows/macOS it starts automatically

# 2. install Python deps
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 3. configure (optional - defaults already work)
cp .env.example .env

# 4. run the API
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> for an interactive UI.

```bash
# upload a PDF
curl -F "file=@mydoc.pdf" http://localhost:8000/upload

# ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Run the tests:

```bash
pytest
```

> On first run the embedding model (~90 MB) downloads automatically and the
> `data/chroma/` folder is created. The Qwen2.5 3B model (~2 GB) is downloaded
> once by `ollama pull`.

---

## Design choices (useful for interviews)

- **Local LLM via Ollama** (`qwen2.5:3b`) – small open-weight model, runs on a
  laptop, no API key or bill. Swap the model by changing `LLM_MODEL` in `.env`.
- **Local embedding model** (`all-MiniLM-L6-v2`) – no API key, no cost, fine for
  learning. Swap for a hosted model later by changing one line in `config.py`.
- **ChromaDB on disk** – zero setup, no server to run. It is a drop-in stand-in
  for a managed vector DB later.
- **No classes / no abstraction layers** – just small named functions in five
  files. Each pipeline step is one function you can point at and explain.
- **Sources returned with every answer** – makes it easy to see *why* the model
  said what it said.

---

## Roadmap (later phases)

Deliberately **not** in phase 1, to be added on top of this same structure:

| Phase | Adds |
|-------|------|
| 2 | PostgreSQL for document metadata & history; Alembic migrations |
| 3 | Docker + `docker-compose` (api + chroma + postgres) |
| 4 | Auth (API keys), background workers for ingestion |
| 5 | CI/CD (GitHub Actions), AWS deployment |
| 6 | Better retrieval (hybrid search, reranking), chat history |
| 7 | Monitoring/logging, RAG evaluation (faithfulness, answer relevance) |

---

## Next implementation step

Fill in the pipeline and run it end to end:

1. `ollama pull qwen2.5:3b`, then `pip install -r requirements.txt` and create `.env`.
2. Confirm `pytest` passes (it should already - the tests don't call Ollama).
3. Start the server and `POST /upload` a small PDF; check the response says
   `chunks_added > 0` and that `data/chroma/` now has files.
4. `POST /ask` a question you know the PDF answers; check `answer` and `sources`.
5. Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` in `.env` and notice how the
   answers change.

Once that works, move to phase 2.
