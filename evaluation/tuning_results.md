# RAG Retrieval Tuning Results

| top_k | Retrieval Hit Rate | Answer Accuracy | Avg. Latency |
|------:|-------------------:|----------------:|-------------:|
| 4 | 75% | 86% | 6.70s |
| 6 | 80% | 90% | 6.55s |
| 8 | 85% | 100% | 6.78s |

Best configuration so far: `top_k = 8`.