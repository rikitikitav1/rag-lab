# 2026-07-28 - Reranking re-measured with the numeric judge (per-k, agent pipeline)

**Why redo it:** the earlier rerank A/Bs (2026-07-25) were categorical. Now that the judge is numeric 0-10 and default k moved to 5, re-measure the reranker as a per-k on/off sweep. Baseline OFF is the numeric agent k-sweep already judged (no need to re-run OFF). Only the reranker toggles; same set `paraphrased_ru` (100), generator `llama3.1:8b`, temp 0, judge `qwen2.5:7b`.

**Plumbing change:** rerank became a per-run knob in the agent pipeline too (previously agent read only the global `rerank.enabled`). `use_rerank` now threads `agent.run -> dispatch -> _search_corpus -> chat.search_chunks`; config stays the default, the per-run value overrides. Removed the "rerank override not supported for agent" guard in `/eval/run` and `/eval/experiment`.

**k=5 data point (single_shot ON):** run `ss_rerank_k5`, faithfulness 7.57, relevance 8.55, completeness 6.27 (100/100). Caveat: this run is single_shot while the OFF baseline and the 3/7/10 ON sweep are agent, so k=5 is not a clean same-pipeline delta. Kept as a cross-pipeline reference, not a comparison row. A clean agent ON k=5 can be added later.

**Agent ON sweep k=3/7/10:** done (`paraphrased_ru_agent_rerank_k{03,07,10}`), 100/100 judged each, backfilled as Experiment id=6 (`rerank_sweep_agent_on`) against the OFF baseline Experiment id=5 (`k_sweep_numeric_baseline`).

| k | faith OFF | faith ON | Δ | rel OFF | rel ON | compl OFF | compl ON | Δ |
|---|-----------|----------|-----|---------|--------|-----------|----------|-----|
| 3 | 6.98 | 6.93 | −0.05 | 8.68 | 8.62 | 5.86 | 6.08 | +0.22 |
| 5 | 7.18 | 7.57 (ss) | (ss) | 8.90 | 8.55 (ss) | 6.16 | 6.27 (ss) | (ss) |
| 7 | 6.78 | 7.03 | +0.25 | 9.06 | 8.99 | 6.06 | 6.46 | +0.40 |
| 10 | 6.99 | 6.97 | −0.02 | 8.94 | 9.15 | 6.09 | 6.28 | +0.19 |

**Conclusion: on the numeric judge rerank does NOT deliver the big faithfulness lift the categorical judge once showed (~+17 faithful count).** Faithfulness is essentially flat (−0.05 / +0.25 / −0.02 at k=3/7/10); the real, consistent gain is on **completeness (+0.19 to +0.40)**, with relevance mixed (best at k=10, +0.21). So the earlier "rerank stacks positively, worth enabling" verdict was largely a categorical-counter artifact: rerank helps, but modestly and on a different axis (completeness) than advertised, not a faithfulness jump. Same lesson as the k-sweep: the categorical ruler exaggerated an effect the numeric ruler sizes correctly. Reranking stays **default OFF** (the ~10s/query CPU cost is not justified by a completeness bump), opt-in for completeness-sensitive workloads.
