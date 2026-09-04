"""Tiny RAG evaluation - no external framework, no paid APIs.

Runs a small set of hand-written questions through the REAL pipeline (the same
`retrieve_chunks` and `answer_question` the API uses, backed by local Ollama)
and prints three numbers: retrieval hit rate, answer accuracy, average latency.

Every run is also recorded as one local MLflow run (experiment "rag-evaluation")
so you can compare different configurations over time.

Before running:
  - the RAG stack must work locally (Ollama running, model pulled, PDFs ingested)
  - edit `eval_dataset.json` so the questions and expected_facts match the
    documents you actually ingested

Run from the project root:
  python -m evaluation.evaluate
"""

import json
import time
from pathlib import Path

import mlflow

from app.config import settings
from app.rag import answer_question, retrieve_chunks

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
EXPERIMENT_NAME = "rag-evaluation"


def load_dataset() -> list[dict]:
    """Read the list of {question, expected_facts} entries from JSON."""
    with DATASET_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def retrieved_text(question: str) -> str:
    """Retrieve chunks for a question and join them into one lowercase string."""
    chunks = retrieve_chunks(question)
    return " ".join(chunk.page_content for chunk in chunks).lower()


def count_facts(text: str, facts: list[str]) -> int:
    """How many expected facts appear (as plain, case-insensitive substrings)."""
    lowered = text.lower()
    return sum(1 for fact in facts if fact.lower() in lowered)


def log_run_params() -> None:
    """Record the RAG configuration used for this evaluation run."""
    mlflow.log_params(
        {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "use_reranker": settings.use_reranker,
            "rerank_top_n": settings.rerank_top_n,
            "reranker_model": settings.reranker_model,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        }
    )


def evaluate() -> None:
    dataset = load_dataset()
    n = len(dataset)

    # One MLflow run per call to this script.
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        log_run_params()

        retrieval_hits = 0     # questions where >= 1 expected fact was in the chunks
        accuracy_sum = 0.0     # sum of "fraction of facts present in the answer"
        latency_sum = 0.0      # sum of answer response times (seconds)

        for item in dataset:
            question = item["question"]
            facts = item["expected_facts"]

            # --- Retrieval Hit Rate: did retrieval surface any expected fact? ---
            hit = count_facts(retrieved_text(question), facts) > 0
            retrieval_hits += int(hit)

            # --- Average response time: time the full answer call ---
            start = time.perf_counter()
            result = answer_question(question)
            elapsed = time.perf_counter() - start
            latency_sum += elapsed

            # --- Answer Accuracy: fraction of expected facts found in the answer ---
            coverage = count_facts(result.answer, facts) / len(facts)
            accuracy_sum += coverage

            print(f"- {question}")
            print(f"    expected fact retrieved : {'yes' if hit else 'no'}")
            print(f"    facts present in answer : {coverage:.0%}   ({elapsed:.2f}s)")

        # Same numbers as before, just kept in variables so we can also log them.
        retrieval_hit_rate = retrieval_hits / n
        answer_accuracy = accuracy_sum / n
        average_latency_seconds = latency_sum / n

        print("=" * 44)
        print(f"Questions          : {n}")
        print(f"Retrieval hit rate : {retrieval_hit_rate:.0%}")
        print(f"Answer accuracy    : {answer_accuracy:.0%}")
        print(f"Average latency    : {average_latency_seconds:.2f}s")

        mlflow.log_metrics(
            {
                "retrieval_hit_rate": retrieval_hit_rate,
                "answer_accuracy": answer_accuracy,
                "average_latency_seconds": average_latency_seconds,
                "number_of_questions": n,
            }
        )


if __name__ == "__main__":
    evaluate()
