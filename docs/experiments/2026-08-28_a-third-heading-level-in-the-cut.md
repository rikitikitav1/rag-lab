# 2026-08-28 - A third heading level in the cut, and what it did not buy

`clean_1024` cuts a file by its `##` headings and falls back to the counter when a section does
not fit. `prefix_1024` inserts one more level before the counter: an oversized section is split by
its `###` headings first, and that third heading rides in the chunk's prefix. The bet was that a
narrower chunk embeds more precisely. This entry is the first point of the ceiling grid, and it
varies the number of levels rather than the ceiling itself.

## Setup

**Set** `paraphrased_v2_ru` (n=823) · **corpus** `prefix_1024` against `clean_1024` · **no judge**:
this is retrieval only, read as the rank of the gold file and of the gold section

What varies: `chunker` (`rooted` → `structured`). What is pinned by construction: ceiling 1024,
`ceiling_on: body`, the same source hygiene, the same question set and the same questions hash,
exact search with a pool of 100. The pair was declared as a one-variable pair when `clean_1024` was
named point zero of the grid.

Code: `app/ingest.py` (`cut_structured`), `scripts/retrieval_report.py` for the measurement and the
paired comparison. Corpus fingerprints: `clean_1024` 12 102 chunks, `prefix_1024` 17 498, both
reproducing from their own code (`scripts/cut_digest.py`, zero differences over 177 sources).

The decision rule was written before the run, in
`~/working_docs/projects/rag-lab/notes/hygiene_arc_log.md`: five predictions with falsification
conditions, and the reporting rule of the grid (the winner is chosen on half A and reported on
half B).

## Result

| level | n | ΔMRR | 95% CI | better / worse / unchanged |
|---|---|---|---|---|
| section, all | 823 | **−0.0109** | [−0.0253, +0.0032] | 133 / 160 / 530 |
| section, half A | 401 | +0.0061 | [−0.0135, +0.0263] | 71 / 76 / 254 |
| section, half B | 422 | −0.0271 | [−0.0466, −0.0077] | 62 / 84 / 276 |
| file, all | 823 | −0.0042 | [−0.0167, +0.0077] | 83 / 105 / 635 |

`hit@1` at section level 0.5298 → 0.5140. Questions whose gold section is not found anywhere in
the pool: 75 → 77 (16 lost, 14 found; sign test ≈ 0.86, no movement).

**There is no winner on half A**, so the pre-registered reading produces no claim. Half B is the
only interval clear of zero and it is negative, but half B alone is not a registered reading.

## What the pre-registration got wrong

Two of five predictions missed, and both misses are about the predictions rather than the corpus.

**A plus was predicted; a minus arrived.** The two arguments were written down before the run and
the one that was not bet on turned out to matter more.

**A threshold was set that noise could not clear.** The auditor's cut asked for the same delta on
the 142 questions whose gold section holds a chunk whose prefix is longer than its body. Prediction:
the lower end of that interval stays above −0.02. Measured: **−0.0232 [−0.0601, +0.0114]**, against
**−0.0083 [−0.0236, +0.0069]** on the remaining 681. The two are indistinguishable, so the cut
neither shows harm nor rules it out.

The threshold was doomed before the run. The standard deviation of the paired deltas is 0.2048, so
the half-width of the interval is 0.034 at n=142 and 0.014 at n=823. A prediction that the lower
end stays above −0.02 fails for almost any point estimate below +0.014. The rule this leaves:
**compute the half-width from a standard deviation already in hand before writing a prediction on
a subset**, and predict an interval, or a point with a tolerance equal to that half-width.

## The mechanism proposed here is not supported

The first reading offered was competition for pool slots: 17 498 chunks against 12 102, pieces of
one section crowding each other out of twenty places. Split by how many chunks the gold section was
cut into, that reading does not hold:

| gold section split into | n | ΔMRR | 95% CI |
|---|---|---|---|
| the same number of chunks | 134 | −0.0009 | [−0.0327, +0.0279] |
| up to twice as many | 603 | −0.0170 | [−0.0330, −0.0002] |
| more than twice | 86 | +0.0165 | [−0.0322, +0.0644] |

Not monotone: the most heavily split sections do not lose at all. Whatever moved the number, it is
not crowding, and the effect is small against noise on every cut of the data tried so far.

## Decision

`clean_1024` stays the served cut and the point of comparison. The third level is not adopted:
there is no measured gain to pay for 45% more rows, a larger index and a slower build.

What is **not** decided by this entry: the ceiling itself. `prefix_1024` varies the number of
levels, and `cap_2048`, `cap_4096` and an unbounded cut vary the ceiling. They remain unmeasured
and this result says nothing about them.

The `prefix_dominates` gate keeps its threshold. It declared 73 of 177 sources broken under this
cut, and this run was the direct test of what it guards; the test came back unable to tell. The
gate is calibration, not law, and the question of moving `sliver_share` into the variant policy
stays open rather than answered by a failed prediction.

## Caveats

- **The rows are gone.** `prefix_1024` was dropped from `data_chunks` after this entry was written,
  because keeping a losing variant indexed changes the plan of every other variant: the crossover
  where the planner abandons the hnsw walk is priced against a read of the whole table. The cut
  reproduces from its own code, so re-indexing restores it in about twenty minutes, and the
  per-question rows survive in `datasets/measurements/`.
- Numbers are exact search, not the index, so they say nothing about what the served path returns
  at any depth.
- One set, one language direction: Russian paraphrases over an English corpus. The same cut may
  behave differently where the question and the corpus share a language.
- Half A and half B disagree in sign. With intervals this wide that is what two halves of one
  measurement look like, not two results.
