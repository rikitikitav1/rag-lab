# 2026-07-28 - Reranking re-measured with the numeric judge (per-k, agent pipeline)

The rerank A/Bs of 2026-07-25 were scored by the categorical judge, which has since been shown to
invert conclusions. With a numeric judge and the default k moved to 5, what does the reranker
actually buy, per k?

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, post-ablation · **judge** `qwen2.5:7b`, numeric 0-10

Baseline off is the numeric agent k-sweep already judged, so only the on arms are new. Generator
`llama3.1:8b`, temp 0, agent pipeline, the reranker the only toggle.

Plumbing changed with it: rerank became a per-run knob in the agent pipeline too (it previously
read only the global `rerank.enabled`), threaded through `agent.run` → `dispatch` →
`_search_corpus` → `chat.search_chunks`, and the "rerank override not supported for agent" guard
was removed from `/eval/run` and `/eval/experiment`.

Agent on sweep: `paraphrased_ru_agent_rerank_k{03,07,10}`, 100 of 100 judged each, backfilled as
Experiment 6 (`rerank_sweep_agent_on`) against the off baseline, Experiment 5
(`k_sweep_numeric_baseline`).

## Result

| k | faith OFF | faith ON | Δ | rel OFF | rel ON | compl OFF | compl ON | Δ |
|---|-----------|----------|-----|---------|--------|-----------|----------|-----|
| 3 | 6.98 | 6.93 | −0.05 | 8.68 | 8.62 | 5.86 | 6.08 | +0.22 |
| 5 | 7.18 | 7.57 (ss) | (ss) | 8.90 | 8.55 (ss) | 6.16 | 6.27 (ss) | (ss) |
| 7 | 6.78 | 7.03 | +0.25 | 9.06 | 8.99 | 6.06 | 6.46 | +0.40 |
| 10 | 6.99 | 6.97 | −0.02 | 8.94 | 9.15 | 6.09 | 6.28 | +0.19 |

On the numeric judge the reranker does not deliver the faithfulness lift the categorical judge
showed (roughly +17 in the faithful count). Faithfulness is flat (−0.05, +0.25, −0.02 at k=3, 7,
10). The consistent gain is on completeness (+0.19 to +0.40), with relevance mixed and best at k=10
(+0.21). The earlier "rerank stacks positively" verdict was largely a counter artifact: the
reranker helps, modestly, and on a different axis than advertised.

## Decision

Reranking stays default off and opt-in, now for completeness-sensitive workloads rather than for
grounding. Roughly 10s per query on CPU does not buy a faithfulness jump.

## Caveats

- the k=5 on row is a single-shot run while the off baseline and the 3/7/10 on sweep are agent, so
  it is a cross-pipeline reference and not a comparison row; a clean agent on k=5 is still missing
- point estimates: these per-k deltas were not tested pairwise here, and
  [the significance entry](2026-07-28_paired-significance-testing-lands-in-the.md) treats them as
  suggestive until they are
