# 2026-07-25 - Reranking (cross-encoder) A/B

**Hypothesis:** a cross-encoder reranker over hybrid retrieval (retrieve-wide → rerank → narrow) improves which chunks reach the generator, most on the cross-lingual set where retrieval is vector-only.

**Change:** `bge-reranker-v2-m3` (CPU, in-process) reranks the top-20 candidates down to top-3. Baseline = hybrid RRF only.

**Setup:** sets `paraphrased_ru` / `paraphrased` (100 each); generator `llama3.1:8b`, temp 0.1 (single run each; noisy). Caveat: temp 0.1 → part of any delta is sampling.

**Result:**

| set | metric | base | rerank |
|-----|--------|------|--------|
| ru | MRR | 0.652 | 0.693 |
| ru | hit@k | 80% | 81% |
| ru | faithful | 37 | 54 |
| ru | unfaithful | 29 | 15 |
| en | MRR | 0.630 | 0.625 |
| en | hit@k | 79% | 76% |
| en | faithful | 61 | 65 |

**Delta:** retrieval hit@k/MRR ~flat (en slightly worse); faithfulness up markedly on ru (+17 faithful, −14 unfaithful), small on en.

**Conclusion:** reranking barely moves the file-level hit@k, but improves chunk ordering within the top-k enough to lift faithfulness on hard cross-lingual queries. Cost ~10s/query on CPU. **Decision: default OFF, opt-in flag** (per-request in `/chat/question`, per-run in `/eval/run`) - the latency is not worth it in the general case, but it is available for cross-lingual/noisy workloads.

---
