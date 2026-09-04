# RAG Retrieval Tuning Results

| Chunk Size | Chunk Overlap | top_k | Retrieval Hit Rate | Answer Accuracy | Avg. Latency |
|-----------:|--------------:|------:|-------------------:|----------------:|-------------:|
| 1000 | 150 | 4 | 75% | 86% | 6.70s |
| 1000 | 150 | 6 | 80% | 90% | 6.55s |
| 1000 | 150 | 8 | 85% | 100% | 6.55s |
| 800 | 120 | 8 | 85% | 97% | 6.89s |

Baseline: `chunk_size=1000`, `chunk_overlap=150`, `top_k=4`.

Best configuration so far: `chunk_size=1000`, `chunk_overlap=150`, `top_k=8`.

## Reranking Experiment

| Configuration | Retrieval Hit Rate | Answer Accuracy | Avg. Latency |
|---|---:|---:|---:|
| No reranking (`1000/150`, `top_k=8`) | 85% | 100% | 6.55s |
| CrossEncoder reranking | 85% | 99% | 7.38s |

Conclusion: reranking did not improve retrieval quality on this evaluation set and slightly reduced answer accuracy while increasing latency. The current best configuration remains `chunk_size=1000`, `chunk_overlap=150`, `top_k=8` without reranking.