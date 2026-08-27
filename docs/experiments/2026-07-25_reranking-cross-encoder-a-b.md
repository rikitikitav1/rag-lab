# 2026-07-25 - Reranking (cross-encoder) A/B

Does a cross-encoder reranker over hybrid retrieval (retrieve-wide → rerank → narrow) improve
which chunks reach the generator, and does it help most on the cross-lingual set where retrieval
is effectively vector-only?

## Setup

**Set** `paraphrased_ru` and `paraphrased` (n=100 each) · **corpus** pre-variant era, `developer-roadmap` still in · **judge** `qwen2.5:7b`, categorical

`bge-reranker-v2-m3` on CPU, in-process, reranks the top 20 candidates down to top 3. Baseline is
hybrid RRF alone. Generator `llama3.1:8b` at temp 0.1, one run per arm.

## Result

| set | metric | base | rerank |
|-----|--------|------|--------|
| ru | MRR | 0.652 | 0.693 |
| ru | hit@k | 80% | 81% |
| ru | faithful | 37 | 54 |
| ru | unfaithful | 29 | 15 |
| en | MRR | 0.630 | 0.625 |
| en | hit@k | 79% | 76% |
| en | faithful | 61 | 65 |

Retrieval hit@k and MRR stay flat, en slightly worse. Faithfulness moves markedly on ru (+17
faithful, −14 unfaithful) and barely on en: reranking does little to which file is found and more to the ordering inside the top k.

## Decision

Default off, opt-in by flag (per-request in `/chat/question`, per-run in `/eval/run`). Roughly 10s
per query on CPU is not worth it in the general case and is available for cross-lingual or noisy
workloads.

## Caveats

- temp 0.1 with one run per arm, so part of every delta is sampling
- counts, not scores; [the numeric re-measurement](2026-07-28_reranking-re-measured-with-the-numeric.md)
  later sized this effect very differently
