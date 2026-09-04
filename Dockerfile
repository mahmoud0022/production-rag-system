# Image for the FastAPI RAG app only.
# Ollama and ChromaDB are NOT in here:
#   - Ollama runs on your Windows host
#   - ChromaDB is just files under /app/data (kept via a bind mount in docker-compose.yml)

FROM python:3.13-slim

# Cleaner Python behaviour inside a container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
# Note: sentence-transformers pulls in PyTorch, so the first build is large and slow.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app ./app

EXPOSE 8000

# Bind to 0.0.0.0 so the port is reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
