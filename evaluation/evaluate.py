"""Tiny RAG evaluation - no external framework, no paid APIs.

Runs a small set of hand-written questions through the REAL pipeline (the same
`retrieve_chunks` and `answer_question` the API uses, backed by local Ollama)
and prints three numbers: retrieval hit rate, answer accuracy, average latency.

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

from app.rag import answer_question, retrieve_chunks

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


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


def evaluate() -> None:
    dataset = load_dataset()

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

    n = len(dataset)
    print("=" * 44)
    print(f"Questions          : {n}")
    print(f"Retrieval hit rate : {retrieval_hits / n:.0%}")
    print(f"Answer accuracy    : {accuracy_sum / n:.0%}")
    print(f"Average latency    : {latency_sum / n:.2f}s")


if __name__ == "__main__":
    evaluate()
