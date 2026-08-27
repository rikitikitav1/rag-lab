# 2026-07-26 - Corpus ablation: disable developer-roadmap

Several `paraphrased_ru` misses look like "right topic, wrong repo": the generalist
`developer-roadmap` source overlaps many domains and its chunks outrank the true gold chunk. None
of the set's gold sources live in roadmap, so switching it off should raise hit@k rather than lower
it.

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, `developer-roadmap` on against off · **judge** none, retrieval only

Runtime source toggle (`PUT /v1/source/3 {active:false}`), no re-index: `hybrid_search` filters
chunks to active sources. Retrieval otherwise identical: `base_ru_t0` (roadmap on) against
`no_roadmap_ru` (roadmap off). Retrieval only, no generation axes.

## Result

| metric | roadmap on | roadmap off | Δ |
|--------|-----------|-------------|---|
| hit@k | 80% | 90% | +10pp |
| MRR | 0.652 | 0.832 | +0.180 |
| misses | 20 | 10 | −10 |

Removing one source lifted hit@k by ten points and MRR by 0.18. Its chunks were pure distractors
ranking above true sources, which confirms the earlier miss diagnosis: disambiguation, not a corpus
gap.

## Decision

Drop `developer-roadmap` from the corpus. It is a roadmap index whose chunks are shallow and
topically diffuse: retrieval noise across every domain, canonical source for none. New default
retrieval on `paraphrased_ru` is 90% / MRR 0.832. The source-active toggle stays as the ablation
tool that justified the removal.

## Caveats

- corpus composition is a first-order retrieval lever here, which also means this number rides on
  a set whose gold labels are single-source
- multi-source gold labels would say how much of the residual distraction is genuinely wrong
  rather than adjacent-correct; they do not exist yet
