# 2026-07-28 - Generator A/B: llama3.1 8b vs 70b (CPU) - does a bigger model earn its cost?

Faithfulness is where the 8B generator is presumed weakest: lightly extrapolated sevens dominate
its distribution. A 70B should extrapolate less. How much quality is the small model actually
costing?

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, post-ablation · **judge** `qwen2.5:7b`, numeric 0-10

Single-shot, k=5, rerank off, temp 0, generator the only variable. The 70B does not fit an 8 GB
card, so Ollama ran it on CPU at about 1.9 tokens per second; the client timeout had to become
tunable (`LLM_TIMEOUT`) because 120s killed the first probe. Retrieval came out identical to the
third digit in both arms (0.93 / 0.845). Backfilled as Experiment 7 (`model_ab_8b_70b`).

## Result

| | 8b | 70b | Δ |
|---|-----|-----|-----|
| faithfulness | 7.22 | **7.60** | +0.38 |
| relevance | 8.55 | 8.51 | −0.04 |
| completeness | 6.19 | 6.09 | −0.10 |
| sec/question | **7.5** | 175.7 | 23x |

The 70B buys +0.38 faithfulness and nothing else, at 23 times the latency: about three minutes per
question on CPU, which is batch-only territory.

Two readings are worth keeping side by side. The cross-encoder reranker on the 8b (`ss_rerank_k5`,
faithfulness 7.57) reproduces almost the whole 70B gain at roughly an eighth of the cost, so "scale
the generator" and "rerank the context" are near-substitutes on this corpus and the reranker is the
cheaper lever. And the equal-weight RRF composite ranks the 8b first (0.04892 against 0.04865): the
8b takes two axes by hair-thin margins, the 70b takes one by a large one, and rank fusion is blind
to magnitude. The composite says no overall win, the per-axis view says one real effect on
faithfulness. Read both, decide from components.

## Decision

Generator stays `llama3.1:8b`; the 70B is deleted. The faithfulness lever of choice remains
reranking (opt-in) or corpus and prompt work, not scale.

## Caveats

- +0.38 is a point estimate here; the paired test arrived a day later and called it borderline
  (p=0.034, CI grazing zero), see
  [the significance entry](2026-07-28_paired-significance-testing-lands-in-the.md)
- the 70B ran on CPU, so the latency ratio is a property of this hardware, not of the model
