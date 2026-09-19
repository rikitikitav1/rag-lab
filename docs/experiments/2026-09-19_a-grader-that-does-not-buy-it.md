# 2026-09-19 - An LLM grader between search and the answer, and the bar it does not clear

A corrective RAG pipeline puts a grader between retrieval and generation: the model reads each
retrieved chunk and says whether it helps answer the question, and the chunks it calls foreign never
reach the generator. This entry records what that node does on our corpus, measured without a judge,
and why it stays off by default.

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

**Nothing from the canon form.** One verdict over the whole retrieved context, as the LangGraph
tutorials write it, drops 0.9% of strangers at "as said" and 14% at the 0.95 cut with retention
0.906. On this corpus it is not a filter, and the per-chunk form is the one measured above.

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

## Reading

The node works, it is better than the cheaper instrument at the retention the bar allows, and it does
not clear the bar this bench set for it. That is a statement about the bar as much as about the
filter: the gold is marked by file, the neighbour class is invisible to both numbers, and 0.95 and
0.50 are round numbers chosen before any measurement. The honest form of the result is the narrow one:
**at this bar, on this corpus, with this model, filtering does not buy enough to be on by default**.

It stays off by default in both paths that can run it, and the switch is a run's own option
(`grade_chunks`). What would change the answer is the measurement this entry does not contain:
whether a filtered context makes the answer better grounded, generation against generation, with the
generator's own floor beside it. Until that runs, "it drops noise" is not a reason to pay four seconds
a question and 2.4% of the answers.
