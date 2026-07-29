# 2026-07-28 - k-sweep re-judged with a numeric 0-10 judge (the categorical verdict reversed)

**Why:** the categorical judge (faithful/partially/unfaithful) buries a good-but-slightly-extrapolated answer ("7") and a mostly-invented one ("3") in the same "partially" bucket. Suspecting the metric hid the truth, replaced it with a numeric 0-10 judge (rubric per axis, `score` via structured output, `judge.py`/`generation_metrics.py` on averages, prompts `judge_*.v2`). Same runs re-judged (no regeneration, only re-scoring). This is a "quality of measurement" fix, not a RAG change: you cannot optimize a system with a lying ruler.

**Numeric curve (0-10, paraphrased_ru 100, agent, temp 0, judge qwen2.5:7b v2):**

| k | faithfulness | relevance | completeness |
|---|-----:|-----:|-----:|
| 1 | 6.79 | 8.60 | 5.48 |
| 3 | 6.98 | 8.68 | 5.86 |
| **5** | **7.18** | 8.90 | **6.16** |
| 6 | 6.83 | 8.87 | 5.99 |
| 7 | 6.78 | **9.06** | 6.06 |
| 10 | 6.99 | 8.94 | 6.09 |

**The numeric judge REVERSED the categorical verdict.** Categorical said faithful peaks at k=3 (41) then collapses to 22 -> "wider retrieval HURTS faithfulness". Numeric shows faithfulness RISES to a peak at k=5 (7.18) then plateaus ~6.8-7.2, it does not collapse. Reason: as k grows, faithful answers drift into "partially" (categorical counter drops), but those are SEVENS (good answers, lightly extrapolated). The average sees it, the counter does not. **The categorical metric lied in the SIGN of the conclusion, not just the magnitude.** Peaks: faithfulness/completeness at k=5, relevance at k=7.

**RRF composite over the 3 axes -> k=5.** Ranking each axis and fusing (RRF, k=60, equal weights): k=5 wins (0.0487), first on faithfulness and completeness, third on relevance. k=10 second (stable #2 on every axis), k=7 third (first on relevance but ranked last on faithfulness, which tanks it). Classic fusion property: stability beats a single peak.

**Weighted RRF / "relevance matters most":** to move the optimum to k=7 relevance needs weight x5 (i.e. drop the other axes); at x3 k=10 wins on stability. If one axis dominates that hard, skip the composite and take argmax(relevance)=k7. Better framing than weighting: constrained optimization. Faithfulness is a FLOOR (grounding is the point of RAG, a fluent but ungrounded answer is the worst case), relevance is the OBJECTIVE. `max relevance s.t. faithfulness >= threshold`. Faithfulness plateaus above k=1, so the threshold passes k=3..10, and among them max relevance is k=7. So the relevance-first view gives k=7 the honest way, not via weight x5. Product-dependent: fintech (grounding critical) keeps the floor high; a general assistant relaxes it.

**Decision: default k=5** (`results_limit` 3 -> 5). Balanced composite optimum, best faithfulness+completeness, relevance only 0.16 below its own peak. Not truncation-driven (agent ~2 hops, context well under the window); the earlier categorical "collapse" was pure metric artifact.

**Meta-lesson (headline):** quality of measurement is a prerequisite to optimization. A metric with too few levels can invert the conclusion, not just blur it. Fixed the ruler (numeric judge) before trusting any k-conclusion.
