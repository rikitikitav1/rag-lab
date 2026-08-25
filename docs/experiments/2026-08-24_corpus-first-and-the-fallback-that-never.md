# 2026-08-24 - Corpus-first, and the fallback that never fired

Question: the agent can now be forced to ask the corpus before it sees any external tool.
Does the trigger for going outside - "the corpus came back empty" - actually catch the cases
where the corpus has nothing to say?

## Setup

20 questions about repositories deliberately outside the corpus (langgraph, fastmcp, pgvector,
celery, kafka, uv and friends), each shaped as "in repository X, how does Y work" so that
DeepWiki has a fair chance of answering. Agent pipeline, llama3.1:8b, judge qwen2.5:7b,
one variable: `fallback_policy`. Completeness is not judged here: out-of-corpus questions have
no reference answer, so only faithfulness and relevance are reported. The set ships in
`questions.tsv` as `out_of_corpus`, so the runs are reproducible from a clean seed.

| run | faithfulness | relevance | answers with an `mcp:` source | `fallback_reason: empty` | s/question |
|-----|--------------|-----------|-------------------------------|---------------------------|------------|
| `agent_choice` (tools visible from hop 1) | 7.05 | 8.25 | 15/20 | 0 | 18.3 |
| `corpus_first` (tools withheld until empty) | 2.60 | 6.05 | 0/20 | 0 | 5.8 |

## The finding: the gate never opened

Zero out of twenty. Hybrid search always returned something above `distance_threshold = 0.55` -
a jest cheatsheet for the React scheduler, redis streams for Kafka partitions - so the emptiness
condition was never true, the external tools never joined the toolbox, and the agent answered
from whatever the corpus handed it. Faithfulness 2.60 is the sound of a model grounding an answer
in documents that have nothing to do with the question.

So the honest reading of the table is *not* "corpus_first is worse". It is: on a corpus of this
shape the empty rule is a null policy, and what the A/B measured is the cost of a gate that
never opens. The policy plumbing works (verified separately: a question about borscht does
return nothing, and the log then shows `fallback_reason: empty` with the remote schemas joining
at the next hop) - it just almost never triggers on questions that *look* technical.

## Why the distance cannot be the gate

Nearest-chunk distance, per question:

| set | p10 | median | p90 |
|-----|-----|--------|-----|
| out-of-corpus (20) | 0.387 | 0.451 | 0.493 |
| in-corpus (100, paraphrased_ru) | 0.325 | 0.394 | 0.444 |

The distributions sit on top of each other. Embedding distance answers "is there anything here
with similar vocabulary", which is exactly the wrong question: a React cheatsheet is lexically
close to a question about React internals and still cannot answer it.

### Correction, measured on 2026-08-25

The heading above is too strong, and the numbers behind it came from twenty questions. On a hundred
per pool the nearest-chunk distance ranks the pools apart with AUC 0.930 (the cross-encoder gets
0.912), and inside the corpus the two flag comparably badly answered questions: the twenty-two
questions the cross-encoder flags below 0.5 average 5.82 faithfulness against 7.42 for the rest, and
the twenty-two most distant ones average 6.00 against 7.37. Spearman against faithfulness is +0.298
and -0.286.

So what failed was not the signal but the rule built on it: "retrieval returned zero rows after the
0.55 cutoff" never fires, because hybrid search almost always returns something under that cutoff.
The two signals also flag different questions (12 of 22 in common), which is why the next experiment
compares three arms: cross-encoder, distance, and either of them.

## Why the cross-encoder can

Best rerank score per question (bge-reranker-v2-m3, sigmoid of the cross-encoder logit, so
"does this chunk answer this question"):

| set | p10 | median | p90 |
|-----|-----|--------|-----|
| out-of-corpus (20) | 0.005 | 0.022 | 0.914 |
| in-corpus (33, curated) | 0.018 | 0.916 | 0.996 |

Two orders of magnitude between the medians, against a distance gap of 0.06. A cut at 0.5 flags
18 of 20 out-of-corpus questions and 8 of 32 in-corpus ones.

Those 8 are the interesting part, because they are not obviously false alarms. Split the
in-corpus run by the same cut and the judge agrees with it:

| in-corpus questions | n | faithfulness | relevance |
|---------------------|---|--------------|-----------|
| top rerank score < 0.5 | 9 | 5.11 | 6.33 |
| top rerank score >= 0.5 | 24 | 6.83 | 9.42 |

The flag fires on the questions the corpus was supposed to answer and answered badly. That is
what a retrieval evaluator is for.

The price is honest too: reranking added roughly 1.4 s per question in this run (5.8 s to 7.2 s),
and the reranker has to be on the card when the generator wants it.

## What this changes

- `weak` stops being a placeholder. The first feature is the cross-encoder score, with a cut
  around 0.5 on this corpus, re-derived per corpus rather than hardcoded - the two out-of-corpus
  questions above the cut are a reminder that the boundary is empirical.
- `corpus_first` stays the default, but paired with the empty rule alone it is indistinguishable
  from corpus-only for anything the corpus cannot serve.
- The full retrieval-evaluator (logistic regression over `min_distance`, `results_count`, top
  rerank score, question length and language) is still the plan. Today's numbers narrow it: the
  cross-encoder score is the feature that carries the signal, and a gate built on distance alone
  would have been wasted work.

Caveat on the judging: faithfulness for an answer built from DeepWiki measures grounding in the
tool output, not truth. The 7.05 for `agent_choice` says the model repeated what the remote
server told it. Whether DeepWiki is right about langgraph is not something this lab measures.
