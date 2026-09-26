# 2026-09-19 - LLM chunk grader before generation: does not clear the bar, off by default

Before the generator sees the retrieved chunks, a second model call reads each chunk against the
question and drops the ones it calls irrelevant. This is the document-grading step that Corrective RAG
(Yan et al., 2024) and Self-RAG (Asai et al., 2023) both contain, in the form LangGraph's archived
tutorials for the two papers share: a prompted yes or no for every retrieved document. LangGraph files
both papers under self-reflective RAG. This entry records what that step does on our corpus, measured
without a judge, and why it stays off by default.

**This is not a reproduction of either paper.** CRAG scores documents with a fine-tuned T5-large
evaluator, turns the scores into one of three actions, searches the web when retrieval looks wrong, and
filters inside documents, strip by strip. Self-RAG trains the generator itself to emit a relevance token
while it decodes. We trained nothing, the stand has no web search, and we drop whole chunks. The result
below says what a prompted 8B filter buys on this corpus, and nothing about CRAG or Self-RAG as
published. One more difference of degree: LangGraph's grader is lenient by design ("keyword(s) or
semantic meaning related to the question" is enough), ours is told to answer no when a passage only
shares words with the question.

## Setup

**Candidates** frozen once and scored: `datasets/candidates/paraphrased_v2_clean_1024_pool20_20260919.json`
(820 English questions) and its Russian twin (823), corpus variant `clean_1024`, 12102 chunks,
embedder `bge-m3` on ollama, pool of 20 per question, cross-encoder scores from
`BAAI/bge-reranker-v2-m3` on vLLM · **grader** `llama3.1:8b` on ollama in its own role, prompt
`grade.chunk` v1, temperature 0, `max_tokens` 16, the verdict through a JSON schema with its
probability from logprobs · **arm** A, the top five by fusion, which is what the serving path
retrieves (reranking is off by default) · **runs** job 2867 (English, 5737 verdicts, 67 min) and 2868
(Russian, 5916 verdicts, 64 min), no generator and no judge anywhere in the measurement.

Three classes per candidate, by the stand's own predicates: the gold section the question came from,
another section of the gold file (a neighbour), and a stranger. Retention is read on the rows whose
gold section reached the arm, the floor on the rows that have a stranger there, and every share is a
row's own.

**The bar, declared before the first run**: keep at least 0.95 of the gold sections and drop at least
half of the strangers, both read on the lower edge of a 10000-draw bootstrap, not on the point. Two
cuts were declared before the closing pass and only these two are read: "as said", the word the
serving node acts on, and 0.7 on the recorded probability.

## Result

| set | cut | retention | lower edge | strangers dropped | lower edge |
|---|---|---|---|---|---|
| English, 658 rows | as said | 0.9757 | **0.9635** | 0.4394 | 0.4028 |
| English, 658 rows | 0.7 | 0.9529 | **0.9362** | 0.5102 | 0.4731 |
| Russian, 616 rows | as said | 0.9805 | 0.9692 | 0.3825 | 0.3499 |
| Russian, 616 rows | 0.7 | 0.9643 | 0.9497 | 0.4726 | 0.4382 |

The corner is taken by neither cut. It is worth seeing how narrowly: at 0.7 on the English set both
sides clear on the points (0.9529 against 0.95 and 0.5102 against 0.50) and neither clears on the
edge. Read on points, this would have been "the filter buys it"; read as declared, it is not.

The filter's own floor, from three passes over the same 200 questions (a second pass in the same
residency with the call order shuffled, a third after a restart of ollama): agreement 0.9887 and
0.9901, and about 0.01 on the numbers above. The half-strangers point at 0.7 sits inside that floor
(0.511, 0.500, 0.498 over the three passes), so "half the strangers go at 0.7" held in one pass of
three.

## What it does buy

**More than a threshold on the cross-encoder.** The threshold was chosen on 200 questions and
reported on the other 620, which it had never seen. At the grader's own retention (0.9739 against
0.9760) it drops 0.4417 of strangers against 0.2080, a paired difference by row of **0.2337, 95%
interval [0.1890, 0.2792]**. The cross-encoder's own floor, taken across a restart of its server, is
zero to four decimals: the top five never changed on 200 pools, the order on three of them. The scope
of the claim is the high-retention end: raising the cross-encoder's threshold, the difference falls to
+0.04 at retention 0.918 and turns against the grader at retention 0.858, which no bar of ours allows.

**Nothing from a stricter prompt.** A second version of `grade.chunk` ("yes only when the passage
states the answer or a fact the answer must contain") moved the point along the same curve rather than
lifting it: at matched retention the difference in strangers dropped is -0.02 with an interval through
zero, and its whole curve tops out at retention 0.912. One rewording is not a proof about the model,
but it is a measured price for a rewrite that reads much stricter to a person.

**Nothing from one verdict over everything.** LangGraph's agentic-rag page grades the whole retrieved
context in one call, on an edge of the graph; their CRAG and Self-RAG tutorials grade document by
document, which is the form measured above. The single verdict drops 0.9% of strangers at "as said" and
14% at the 0.95 cut with retention 0.906: on this corpus it is not a filter.

## What it costs, and the language it was tuned in

About a second a verdict, so four to five seconds a question at five chunks. On the English set at
"as said" it removes 30.3% of a row's chunks (3.49 of five survive) and leaves 20 rows of 820 with no
context at all, which are answers turned into refusals.

The same grader is milder on Russian questions, and that is a paired measurement rather than an
impression: the two sets pair through the source question, and on 4048 chunks graded under both
wordings the verdicts agree on 0.8654 of them, with **458 flips from no to yes against 87 the other
way**, 259 of them on neighbours of the gold section. Against an engine floor near 0.99, that is an
order of magnitude more disagreement than noise, and it is one-sided. The practical reading: a filter
tuned and judged on the English set does less work on Russian questions (0.38 against 0.44 of
strangers) at slightly higher retention, so these numbers do not transfer across the language of the
question.

## What "unrelated" turned out to mean

The floor counts chunks that came from a file the question was not marked against. That label is
mechanical, and this corpus is a set of interview-question READMEs whose topics overlap, so it was
worth asking what the label holds. A hundred of those chunks were drawn by seed and shown to a much
stronger model (Opus) with the question beside them and nothing else: no verdict of ours, no class, not
even the fact that all hundred came from one class. It called 48 of them helpful on a loose test
(could a competent writer answer from this passage). The same hundred was labelled a second time
against a stricter instruction, copied from the grader's own prompt and adding that a passage about
another technology is not an answer: **39**. The two instructions agree on 91 of the hundred, so the
share of "unrelated" chunks that are on the subject is **between 0.39 and 0.48**, and neither end is
anchored to a person.

Two instruments that are not language models sort the same hundred the same way. The cross-encoder
scores the "helpful" ones at 0.363 against 0.076 for the rest, intervals apart. The heading of the
chunk shares a word with the heading the question came from for 58% of the helpful against 23% of the
rest, with no model in the loop at all. The direction is a property of this corpus: one family of
interview-question repositories whose subjects overlap, so a chunk from an unmarked file is often on
the subject anyway.

Part of the looseness is ours rather than the labeller's. The questions are paraphrases, and the
paraphrase often drops the technology: 28 of those 48 questions contain no word of their own
repository's name. "A variable holding no value against one not assigned yet" no longer says
JavaScript, so a passage about Go honestly helps a blind reader and would ground an answer in the
wrong language. The grader and the generator read the same stripped question.

Crossed with our grader on the same hundred: it kept 36 of the 48 and dropped 38 of the 52, agreeing
with the blind reader on 74 of 100. Agreement between two language models is agreement, not accuracy,
and it is worth exactly that much. Against the stricter labels the same grader keeps 31 of 39 and
drops 42 of 61, agreeing on 73 of 100, so the reading does not depend on which instruction was used.
What it supports: the share of "unrelated" chunks that can be dropped without losing something useful
is between 0.52 and 0.61, so the 0.44 the grader drops is a large part of what was available, and the
floor of 0.50 was demanding rather than mistaken.

## Reading

The node works, it is better than the cheaper instrument at the retention the bar allows, and it does
not clear the bar this bench set for it. That is a statement about the bar as much as about the
filter, and the calibration above says how much: the gold is marked by file, the neighbour class is
invisible to both numbers, 0.95 and 0.50 are round numbers chosen before any measurement, and a
substantial share of what the floor counts as noise is on the subject of the question. The honest form of the result is the narrow one:
**at this bar, on this corpus, with this model, filtering does not buy enough to be on by default**.

It stays off by default in both paths that can run it, and the switch is a run's own option
(`grade_chunks`).

## And the answers themselves

The obvious follow-up was run the same day: a hundred questions drawn by seed from the 620 nobody had
chosen anything on, answered four times through the direct path, twice with the filter and twice
without, all four arms judged by one judge in one residency. Paired by question, the filter moved
groundedness by **+0.01** (interval -0.35 to +0.36) in one pair and **+0.16** (-0.27 to +0.61) in the
other. Both intervals cross zero, and both deltas are smaller than what the same arm scores against
itself when it is simply run twice: **0.18** on groundedness between the two control runs. At a
hundred pairs the interval is about a third of a judge point wide, so this is "no effect of this size
is visible here", not "no effect".

Relevance and completeness came out lower with the filter in both pairs, which looked like a cost of
the shorter context until the rows were split by how many chunks the filter had actually removed. The
rows it left untouched lost as much as the rows it cut (-0.29 against -0.37, both intervals crossing
zero, and the rows that lost the most chunks lost the least score). With an identical context and the
generator at temperature 0.1, that difference is the generator and the judge, not the filter.

The filter costs 4.4 seconds of grading a question. The shorter context does make generation itself
about 0.4 seconds faster, and one row in a hundred ends with no context at all and refuses. So the
answer to the question in the title, on this corpus and with this model: it drops noise, it does not
buy grounding, and it is paid for in seconds and in one refusal per hundred.
