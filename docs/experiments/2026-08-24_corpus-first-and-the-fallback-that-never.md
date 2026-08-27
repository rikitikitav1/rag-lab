# 2026-08-24 - Corpus-first, and the fallback that never fired

The agent can now be forced to ask the corpus before it sees any external tool. Does the trigger
for going outside, "the corpus came back empty", actually catch the cases where the corpus has
nothing to say?

## Setup

**Set** `out_of_corpus` (n=20), with `paraphrased_ru` (n=100) and `curated` (n=33) as reference pools · **corpus** pre-variant era, post-ablation · **judge** `qwen2.5:7b`, numeric 0-10

Twenty questions about repositories deliberately outside the corpus (langgraph, fastmcp, pgvector,
celery, kafka, uv and friends), each shaped as "in repository X, how does Y work" so DeepWiki has a
fair chance. Agent pipeline, `llama3.1:8b`, one variable: `fallback_policy`. Completeness is not
judged, because out-of-corpus questions have no reference answer. The set ships in `questions.tsv`,
so the runs rebuild from a clean seed.

## Result

| run | faithfulness | relevance | answers with an `mcp:` source | `fallback_reason: empty` | s/question |
|-----|--------------|-----------|-------------------------------|---------------------------|------------|
| `agent_choice` (tools visible from hop 1) | 7.05 | 8.25 | 15/20 | 0 | 18.3 |
| `corpus_first` (tools withheld until empty) | 2.60 | 6.05 | 0/20 | 0 | 5.8 |

Zero out of twenty. Hybrid search always returned something above `distance_threshold = 0.55`, a
jest cheatsheet for the React scheduler, redis streams for Kafka partitions, so the emptiness
condition was never true, the external tools never joined the toolbox, and the agent answered from
whatever the corpus handed it. Faithfulness 2.60 is the sound of a model grounding an answer in
documents that have nothing to do with the question.

The honest reading of the table is not "corpus_first is worse". It is that on a corpus of this
shape the empty rule is a null policy, and what the A/B measured is the cost of a gate that never
opens. The plumbing works, verified separately: a question about borscht does return nothing, and
the log then shows `fallback_reason: empty` with the remote schemas joining at the next hop.

## Why the distance cannot be the gate

Nearest-chunk distance, per question:

| set | n | p10 | median | p90 |
|-----|---|-----|--------|-----|
| out-of-corpus | 20 | 0.387 | 0.451 | 0.493 |
| in-corpus (`paraphrased_ru`) | 100 | 0.325 | 0.394 | 0.444 |

The distributions sit on top of each other. Embedding distance answers "is there anything here with
similar vocabulary", which is the wrong question: a React cheatsheet is lexically close to a
question about React internals and still cannot answer it.

### Correction, measured on 2026-08-25

The heading above is too strong, and the numbers behind it came from twenty questions. On a hundred
per pool the nearest-chunk distance ranks the pools apart with AUC 0.930 against the cross-encoder's
0.912, and inside the corpus the two flag comparably badly answered questions: the 22 questions the
cross-encoder flags below 0.5 average 5.82 faithfulness against 7.42 for the rest, and the 22 most
distant ones average 6.00 against 7.37. Spearman against faithfulness is +0.298 and -0.286.

So what failed was not the signal but the rule built on it: "retrieval returned zero rows after the
0.55 cutoff" never fires. The two signals also flag different questions (12 of 22 in common), which
is why the next experiment compares three arms: cross-encoder, distance, and either of them.

## Why the cross-encoder can

Best rerank score per question (`bge-reranker-v2-m3`, sigmoid of the logit, so "does this chunk
answer this question"):

| set | n | p10 | median | p90 |
|-----|---|-----|--------|-----|
| out-of-corpus | 20 | 0.005 | 0.022 | 0.914 |
| in-corpus (`curated`) | 33 | 0.018 | 0.916 | 0.996 |

Two orders of magnitude between the medians, against a distance gap of 0.06. A cut at 0.5 flags 18
of 20 out-of-corpus questions and 8 of 32 in-corpus ones.

Those eight are the interesting part, because they are not obviously false alarms. Split the
in-corpus run by the same cut and the judge agrees with it:

| in-corpus questions | n | faithfulness | relevance |
|---------------------|---|--------------|-----------|
| top rerank score < 0.5 | 9 | 5.11 | 6.33 |
| top rerank score >= 0.5 | 24 | 6.83 | 9.42 |

The flag fires on the questions the corpus was supposed to answer and answered badly, which is what
a retrieval evaluator is for. The price is 1.4s per question in this run (5.8s to 7.2s), and the
reranker has to be on the card when the generator wants it.

## Decision

`weak` stops being a placeholder: the first feature is the cross-encoder score with a cut around
0.5 on this corpus, re-derived per corpus rather than hardcoded. `corpus_first` stays the default,
but paired with the empty rule alone it is indistinguishable from corpus-only for anything the
corpus cannot serve. The full retrieval evaluator (logistic regression over `min_distance`,
`results_count`, top rerank score, question length and language) stays on the plan, narrowed by
today's numbers: the cross-encoder score carries the signal, and a gate built on distance alone
would have been wasted work.

## Caveats

- n=20 on the primary pool: the distance table above did not survive a hundred per pool, see the
  correction
- faithfulness for an answer built from DeepWiki measures grounding in the tool output, not truth
- the two out-of-corpus questions above the 0.5 cut are a reminder that the boundary is empirical
