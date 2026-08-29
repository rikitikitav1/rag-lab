# 2026-07-25 - Reranking on top of v3 (stacking, clean)

The earlier rerank A/B ran at temp 0.1 and was noisy. Does the reranker still pay once the v3
prompt is in place and the temperature is pinned to 0?

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, `developer-roadmap` still in · **judge** `qwen2.5:7b`, categorical

v3 prompt, temp 0. `v3_ru` (rerank off) against `v3_rerank_ru` (rerank on, top 20 → top 3). The
reranker is the only toggle.

## Result

| metric | v3 | v3 + rerank | Δ |
|--------|----|-------------|---|
| MRR | 0.652 | 0.685 | +0.033 |
| hit@k | 80% | 80% | = |
| relevant | 57 | 61 | +4 |
| irrelevant | 7 | 4 | −3 |
| faithful | 50 | 53 | +3 |
| unfaithful | 24 | 17 | −7 |

Everything moves the same way: ranking, grounding, relevance. Modest and consistent, unlike the
temp-0.1 run.

## Decision

Confirms opt-in, default off. The gain is real on the cross-lingual set and does not justify 10s
per query for general use.

## Caveats

- counts, not scores, on a single run per arm and no interval
- [the numeric re-measurement](2026-07-28_reranking-re-measured-with-the-numeric.md) found the
  faithfulness part of this verdict to be a counter artifact: the durable gain is on completeness
