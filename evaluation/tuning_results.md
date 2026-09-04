# RAG Retrieval Tuning Results

| Chunk Size | Chunk Overlap | top_k | Retrieval Hit Rate | Answer Accuracy | Avg. Latency |
|-----------:|--------------:|------:|-------------------:|----------------:|-------------:|
| 1000 | 150 | 4 | 75% | 86% | 6.70s |
| 1000 | 150 | 6 | 80% | 90% | 6.55s |
| 1000 | 150 | 8 | 85% | 100% | 6.55s |
| 800 | 120 | 8 | 85% | 97% | 6.89s |

Baseline: `chunk_size=1000`, `chunk_overlap=150`, `top_k=4`.

Best configuration so far: `chunk_size=1000`, `chunk_overlap=150`, `top_k=8`.