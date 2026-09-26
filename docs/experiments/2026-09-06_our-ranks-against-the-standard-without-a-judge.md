# 2026-09-06 - RAGAS ID-based retrieval metrics against our hit@k and MRR

The first sub-phase of the standard arc asks a narrow question: where RAGAS measures retrieval
without an LLM, does it agree with the `hit@k` and MRR this stand has been reporting for two
months. The answer had to be declared before it was computed, because one of the three comparisons
is an identity by construction, and a report that discovers an identity reads like a finding.

## Setup

**Rows** every archived answer carrying a marked source and a non-empty retrieval **as
`retrieval_metrics.retrieved_sources` defines it**, that is the row's sources minus anything
prefixed `mcp:`, plus the files the gate dropped: n=14,569 across 99 runs, both pipelines. Counting
the raw column instead gives 14,574 and 97 runs, and the difference is that filter · **corpus** whatever each run recorded, this is a comparison of two
readings of the same rows, not of two corpora · **judge** none, no model is called at all

- our side: `hit@k` and the reciprocal rank per row, from `evals/retrieval_metrics`, schema 3
- the standard's side: `IDBasedContextRecall` and `IDBasedContextPrecision`, ragas 0.4.3
- retrieved paths are reduced to the label they contain before being handed over: `is_gold` here
  is substring containment, ragas compares ids by equality, and without the reduction every recall
  would be zero by construction
- in-hop axes do not take part: ragas has no notion of a hop
- numbers from `datasets/measurements/ragas_id_correlation_20260906.json`, reproduced by
  `scripts/ragas_id_correlation.py`

## What was declared before the run

With one label per question `IDBasedContextRecall` is `hit@k` row by row: both answer "was any
expected file retrieved", and neither reads the rank. So a Spearman between them is 1 by
construction, and it measures the wiring, not the instrument.

## Result

```
n = 14,569 rows, 99 runs, agent and single_shot

means        hit@k 0.9016   RR 0.8074   id_recall 0.8958   id_precision 0.5390

identity     one-label rows            14,393 of 14,569  (98.8%)
             rows where they differ         0

spearman     hit@k vs id_recall        0.9569
             MRR   vs id_recall        0.6386
             MRR   vs id_precision     0.6270
             hit@k vs id_precision     0.5312
```

The identity holds exactly where it was predicted to hold: on all 14,393 single-label rows, not
one differs. It does not reach 1.0 overall only because 176 rows carry more than one marked
source, and there the two part ways on purpose: `hit@k` says "at least one expected file came
back", `id_recall` says "this share of them did".

Those 176 rows come from a single set. Multi-label questions exist only in `curated`, 14 of the
30 questions there that carry a marked source at all (33 in the set);
`interview`, both `paraphrased_v2` sets, both veto sets and both small `paraphrased` sets are
single-label throughout. So the exception is one hand-made set of thirty questions, not a property
of the corpus.

## What is actually new

`IDBasedContextPrecision` is an axis this stand does not have. Read from its source rather than its
name: it takes both id lists as **sets**, so it is the share of *distinct* ids retrieved that are
gold, not the share of chunks. That reading matches how a row is stored here, because
`chat.take_sources` already keys by path and collapses several chunks of one file into one entry.

So on our rows it answers: of the files retrieval reached, what share were files the question was
written from. Reached, not shown: the population is the one `hit@k` and MRR rank over, which
includes files the coverage gate hid from the model, because retrieval did find them. Mean 0.539 over 14,569 rows, against a mean of 2.36 distinct
files surfaced per question. Typically one right file among two or three shown.

It correlates with `hit@k` at 0.53 and with MRR at 0.63, which is what a genuinely different
question looks like next to a familiar one.

It is not a partner for MRR, and neither is recall. Verified on synthetic rows before touching the
archive: moving the gold from first place to last leaves both ID metrics unchanged (recall 1.0,
precision 0.2 either way). **Both are rank-blind.** Among the judge-free metrics of the standard
there is no counterpart to MRR at all, and any comparison that pairs them is comparing a rank with
a set.

## What the move to the standard cost

ragas 0.4.3 does not import as its own dependency resolution installs it: it reaches for
`langchain_community.chat_models.vertexai`, which `langchain-community` 0.4.2 no longer has. It
runs with `langchain-community<0.4` (0.3.31), a package whose own maintainers declare it sunset.
The version is pinned whole in the `eval` dependency group, off the runtime image, because the
metric definitions and prompts are the instrument rather than a detail of it.

## Reading

On the half of RAGAS that needs no judge, the standard and this stand agree, and agreement was
the expected outcome rather than a discovery: for 98.8% of rows the two metrics are the same
arithmetic under two names. That is worth having written down, because it turns "we also ran
RAGAS" from a claim into a checked statement, and because it means the ranks reported in every
earlier entry are the ranks the standard would have reported.

The one thing gained is an axis, not a verdict: `id_precision` says how much of what retrieval
shows is off-target, and that number, 0.539, has never appeared in this journal before. It is
coarse by construction, because with 2.36 files per question it can only take values like 1/1,
1/2, 1/3 and 2/3, and it falls when retrieval surfaces more files as well as when it surfaces
wrong ones. It reads as focus, not as accuracy.

By the owner's decision the axis was kept, and kept as ours: `file_precision` in
`evals/retrieval_metrics`, next to `hit@k` and MRR, schema 4. The formula is two set operations
read out of ragas's own source and checked against it, so carrying the library into the runtime
image to compute it would buy nothing. On the two arms of experiment 41 it already reads differently from the old two: `clean_1024` 0.658
against `baseline` 0.580, while `hit@k` slightly favours baseline and MRR is level. Those two
numbers are a recomputation, not a record: experiment 41 was aggregated under schema 3, before the
axis existed, so they live in `datasets/measurements/file_precision_experiment41_20260906.json`
with that stated. n=60 per arm, one pair, no interval claimed.

## The same comparison one grain finer

The file is a coarse gold: a chunk from the right file that sits under the wrong heading counts as a
hit. The `chunks` column added the same day carries each chunk's address, so the comparison was run
again with the identifier as the pair (source, section), on `arc3_agent_baseline`, n=100 in-corpus
rows. Only rows written after 2026-09-06 can be read this way, and the archive never will be.

```
file level      hit@k 0.94   MRR 0.831   id_precision 0.539
section level         0.74         0.603              0.229
```

Twenty points of the hit rate are "right file, wrong section". That is the known weakness of a
file-level gold, and it now has a number.

What did **not** happen is the thing this run was expected to show. The prediction, written into the
checklist beforehand, was that at section level ID-recall would stop being an identity with
`hit@k`. It did not: Spearman 1.0 again, 0 rows of 100 differing.

The reason is simpler than either grain: **the identity is a property of single-label gold, not of
file-level gold.** One gold identifier of any size makes recall binary, and binary recall is the hit.
Only a question with several gold identifiers breaks it, and this corpus has 176 such rows out of
14,569, all in one hand-made set.

So the judge-free half of the standard gives no independent yardstick at any grain while the gold
carries one label. What it gives is one axis this stand did not have, and the confirmation that the
ranks reported here are the ranks the standard would report.

The population is worth stating beside the axis: the gold heading resolves for every set except the
hand-written `curated` (`interview` 60/60, `paraphrased_v2_ru` 60/60, `veto_headings` 60/60,
`paraphrased_ru` 60/60, `curated` 0/30, checked by calling `section_exists` rather than by
re-deriving it in SQL).

The half that could disagree is the judged half, and it has not run: it needs `contexts` on the
row, and only 241 rows of 17,867 carry them. That is the next sub-phase, not this one.
