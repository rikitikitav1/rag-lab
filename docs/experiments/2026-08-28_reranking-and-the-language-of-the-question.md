# 2026-08-28 - Reranking by question language: helps Russian, hurts English on `baseline` only, no language switch

The same cross-encoder helped section ranking on Russian questions and hurt it on English ones, on
one corpus with the same gold sections. A switch by language suggests itself. This entry records the
two signs, what the gap between them decomposes into, and why the switch that follows from it is not
the language. Written on 14.09 from records kept on the stand since August.

## Setup

**Set** `paraphrased_v2_ru` (n=823, Russian paraphrases of interview questions over an English
corpus) and `paraphrased_v2` (n=820, the same questions paraphrased in English) · **corpus**
`baseline` and `clean_1024` · retrieval only, no generator and no judge: exact search, vector and
full-text legs of 100 candidates each merged by RRF, distance threshold off · **reranker**
`bge-reranker-v2-m3` over the top 20 · experiments 12 (Russian) and 13 (English), kind `retrieval`,
paired by question; recorded again as 15 and 16, which matched to the fourth digit with their
intervals

What varies: `rerank_top` 0 against 20, crossed with the corpus variant. The measure is section
MRR@20: the gold section, stricter than the gold file.

## Result

| set | variant | section MRR, plain → rerank | Δ [95% CI] | better / worse / same |
|---|---|---|---|---|
| `paraphrased_v2_ru`, n=823 | `baseline` | 0.599 → 0.636 | +0.0369 [+0.0155, +0.0596] | 214 / 149 / 460 |
| `paraphrased_v2`, n=820 | `baseline` | 0.660 → 0.635 | -0.0245 [-0.0460, -0.0021] | 154 / 191 / 475 |
| `paraphrased_v2_ru`, n=823 | `clean_1024` | 0.642 → 0.688 | +0.0454 [+0.0250, +0.0670] | 190 / 135 / 498 |
| `paraphrased_v2`, n=820 | `clean_1024` | 0.683 → 0.675 | -0.0083 [-0.0283, +0.0125] | 135 / 161 / 524 |

At the file level the signs agree with the table: Russian +0.0371 on `baseline` and +0.0385 on
`clean_1024`, English -0.0018 and +0.0063, both English intervals through zero.

On the corpus served today, `clean_1024`, the English sign is no longer readable: the interval
crosses zero. "Reranking hurts English questions" holds on `baseline` only.

## The gap is not the language

Split by the rank the gold section had before reranking (`baseline`, from the rows of the records,
no new run):

| rank before | Russian Δ, n | English Δ, n |
|---|---|---|
| 1 | -0.1200 [-0.1456, -0.0960], 396 | -0.1578 [-0.1835, -0.1319], 450 |
| 2-3 | +0.1856 [+0.1349, +0.2338], 157 | +0.1582 [+0.1055, +0.2097], 148 |
| 4-10 | +0.2732 [+0.2222, +0.3345], 131 | +0.1976 [+0.1412, +0.2547], 122 |
| 11-20 | +0.2547 [+0.1720, +0.3483], 51 | +0.1123 [+0.0382, +0.2030], 30 |
| not found | 0, 88 | 0, 70 |

The shape is the same on both sets. Reranking permutes the pool: where fusion already put the gold
section first, any move is a loss, and it pulls nothing out of "not found". A delta per bucket proves
little on its own, because a rank-1 bucket can only lose by construction, so the claim rests on
frequencies instead:

| rank before | Russian: dropped / lifted | English: dropped / lifted |
|---|---|---|
| 1 | 0.197 / 0 | 0.247 / 0 |
| 2-3 | 0.255 / 0.516 | 0.291 / 0.446 |
| 4-10 | 0.229 / 0.725 | 0.287 / 0.598 |
| 11-20 | 0.020 / 0.745 | 0.067 / 0.500 |

Two things make the gap of 0.0614. English questions have the gold section first more often (0.549
against 0.481), so more of them sit in the bucket that can only lose; substituting one set's
composition into the other's deltas explains about a third of the gap (0.0172 and 0.0193). The rest
is the reranker itself: on English questions it drops more often and lifts less often in every
bucket. The Russian questions are translations of the same originals with the same gold sections,
so the difference lives in the pair of question and candidate, that is, in the model. Why the model
behaves so, this bench cannot say.

## The switch that follows, and why it was not built

If the loss lives where fusion is already right, the switch belongs on fusion's confidence, not on
the language. Without reranking, on `clean_1024`, the distance to the first candidate predicts
whether that candidate is right: on the Russian set the best decile is right 0.817 of the time and
the worst about 0.28. Per question values are in
`datasets/measurements/confidence_calibration_paraphrased_v2_ru_clean_1024.json` and
`confidence_calibration_paraphrased_v2_clean_1024.json`.

The gate built on that signal has its own entry,
[a gate that degenerated the other way](2026-08-28_a-gate-that-degenerated-the-other-way.md). In
short: on Russian it loses to reranking everything (on the reported half, +0.0179 [-0.0089, +0.0451]
against +0.0294 [+0.0001, +0.0595] for unconditional reranking on the same half), and on English it wins only by reranking 2% of the
questions, which is what the default already does.

## Decision

- No switch by language and no conditional reranking. The idea was closed on 28.08, not deferred.
- Reranking stays off by default, for another reason: the 8b generator the agent needs takes the
  card room the reranker held (`config.yaml`, and the
  [generator grid](2026-08-29_generator-grid-4b-against-8b.md)). The +0.0454 on Russian is what that
  costs in ranking.
- Because the sign depended on the set, later corpus comparisons carry the English set as a second
  voice: a variant that wins on one language by losing on the other is not taken.

## Caveats

- Retrieval only. What reranking does to answers depends on the generator, and is in the generator
  grid entry linked above.
- The buckets and frequencies were read after the fact from rows already recorded, and on `baseline`
  only; they were not repeated on `clean_1024`.
- "Language" here means a Russian paraphrase over an English corpus against an English paraphrase of
  the same question: cross-lingual against same-language retrieval, not Russian against English in
  general.
- One reranker over the top 20, one embedder (`bge-m3`).
