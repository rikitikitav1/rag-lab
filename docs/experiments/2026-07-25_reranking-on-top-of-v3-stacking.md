# 2026-07-25 - Reranking on top of v3 (stacking, clean)

**Hypothesis:** does the cross-encoder reranker stack on the v3 prompt? The earlier rerank A/B was temp 0.1 (noisy); redo cleanly on v3.

**Setup:** set `paraphrased_ru` (100); v3 prompt, **temp 0**; `v3_ru` (rerank off) vs `v3_rerank_ru` (rerank on, top-20 → top-3). Only the reranker toggled.

**Result:**

| metric | v3 | v3 + rerank | Δ |
|--------|----|-------------|---|
| MRR | 0.652 | 0.685 | +0.033 |
| hit@k | 80% | 80% | = |
| relevant | 57 | 61 | +4 |
| irrelevant | 7 | 4 | −3 |
| faithful | 50 | 53 | +3 |
| unfaithful | 24 | 17 | −7 |

**Delta:** everything moves the right way - better ranking (MRR), better grounding (unfaithful −7), better relevance (+4). Modest but consistent, unlike the noisy temp-0.1 run.

**Conclusion:** rerank **stacks positively on v3** on the cross-lingual set, at ~10s/query CPU cost. Confirms the **opt-in default OFF** decision: the quality gain is real and worth enabling for cross-lingual / high-stakes queries, not for latency-sensitive general use.

---
