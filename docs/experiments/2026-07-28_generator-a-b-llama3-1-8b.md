# 2026-07-28 - Generator A/B: llama3.1 8b vs 70b (CPU) - does a bigger model earn its cost?

**Hypothesis:** faithfulness is the axis where the 8B generator is presumed weakest (lightly-extrapolated "sevens" dominate the distribution). A 70B generator should extrapolate less. Question: how much quality is the small model actually costing us?

**Setup:** single_shot, k=5, rerank off, temp 0, set `paraphrased_ru` (100), judge `qwen2.5:7b` fixed. The 70B does not fit an 8 GB GPU, so Ollama ran it on CPU (~1.9 tok/s; the client timeout had to become tunable - `LLM_TIMEOUT` env - because 120s killed the first probe). Same questions, same retrieval: hit@k and MRR came out **identical to the third digit** (0.93 / 0.845), confirming the model is the only variable. Backfilled as Experiment id=7 (`model_ab_8b_70b`).

| | 8b | 70b | Δ |
|---|-----|-----|-----|
| faithfulness | 7.22 | **7.60** | +0.38 |
| relevance | 8.55 | 8.51 | −0.04 |
| completeness | 6.19 | 6.09 | −0.10 |
| sec/question | **7.5** | 175.7 | 23x |

**Conclusion:** the 70B buys **+0.38 faithfulness and nothing else, at 23x the latency** (~3 min/question on CPU: batch-only territory, unusable interactively). Two framings make this interesting:
- The cross-encoder reranker on the 8b (`ss_rerank_k5`: faith 7.57) reproduces almost the entire 70B faithfulness gain at ~1/8 of the cost. "Scale the generator" and "rerank the context" turn out to be near-substitutes on this corpus, and the reranker is the cheaper lever.
- The equal-weight RRF composite actually ranks **8b first** (0.04892 vs 0.04865): 8b takes two axes by hair-thin margins, 70b takes one by a large one, and rank fusion is blind to magnitude. A composite says "no overall win"; the per-axis view says "one real effect, on faithfulness". Read both, decide from components.

Pending: the +0.38 is a point estimate; paired CI / significance testing (in progress as an aggregator extension) should confirm it before it hardens into a claim. **Decision: generator stays llama3.1:8b.** The 70B is deleted; the faithfulness lever of choice remains reranking (opt-in) or corpus/prompt work, not scale.
