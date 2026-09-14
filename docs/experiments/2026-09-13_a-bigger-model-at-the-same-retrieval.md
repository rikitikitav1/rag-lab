# 2026-09-13 - A bigger model at the same retrieval, and where it refuses

The owner's question: what does a large cloud model buy over the 8b on this RAG when retrieval is
held the same? Not the cloud against the stand: the same questions, the same retrieved context and
the same judge, with only the generator changed, and each arm run twice so a difference between the
models can be read against the difference between two runs of one model.

## Setup

**Set** the 300 questions of `arc3_agent_baseline`: 100 `paraphrased_ru` from the corpus, 100
`out_of_corpus` (87 English, 13 Russian), 100 `off_domain` (76 Russian, 24 English) · **corpus**
`clean_1024`, `single_shot`, no reranking, embedder `bge-m3` on ollama · **generators**
`llama3.1:8b` on ollama on the card and `deepseek-ai/DeepSeek-V4-Flash-0731` through the gonka
broker, `temperature` 0.1, `max_tokens` 1024 · **judge** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM,
JSON without free whitespace, all four arms judged in one residency (the rejudged copies 55-58) ·
experiments 52 (repeat A) and 53 (repeat B), kind `generation`, each holding both arms

The predictions were written before the run in the arc's log. The grid's floor is its pair of
repeats. The cloud model's own floor was measured first: two passes over 50 other questions, under
two cache keys, because the broker returns an exact repeat of a request body from its cache.

## The cloud's floor

Two passes of DeepSeek over the same 50 questions (jobs 2780 and 2781): 0 of 50 answers equal byte
for byte; verdicts moved on faithfulness 18 of 44 (mean absolute difference 1.50), on relevance 21 of
44 (0.91), on completeness 2 of 33 (0.18); no 429 and no 5xx. Both passes were judged by one judge
under the older JSON rule, so the pair is comparable with itself.

## Result on the corpus pool

| repeat | axis | llama | DeepSeek | Δ [95% CI] | p |
|---|---|---|---|---|---|
| A, n=100 | faithfulness | 8.21 | 8.18 | -0.03 [-0.50, +0.45] | 0.91 |
| A, n=100 | relevance | 8.69 | 7.61 | -1.08 [-1.73, -0.47] | 0.0049 |
| A, n=100 | completeness | 6.46 | 6.05 | -0.41 [-0.81, -0.03] | 0.022 |
| B, n=99 | faithfulness | 8.22 | 8.12 | -0.10 [-0.62, +0.37] | 0.89 |
| B, n=99 | relevance | 8.88 | 8.02 | -0.86 [-1.46, -0.29] | 0.011 |
| B, n=99 | completeness | 6.63 | 6.26 | -0.36 [-0.78, +0.04] | 0.069 |

Paired over the same questions. Under Holm over the six tests only relevance in A passes (0.0049
against 0.0083); relevance in B misses by a hair (0.0106 against 0.01), completeness passes in
neither. The sign is the same in both repeats on every axis.

Against the repeats: the relevance means of two runs of one model differ by 0.17 (llama) and 0.41
(DeepSeek), below the models' gap of 1.08 and 0.86. Row by row the repeats are far apart: at
`temperature` 0.1 only 4% of answers repeat byte for byte, and a row's verdict moves by about a point
between repeats (mean absolute difference llama 0.93 / 1.07 / 0.62, DeepSeek 1.29 / 1.42 / 0.60 on
faithfulness, relevance, completeness). So means over 100 rows are readable and single rows are not.

## Where the relevance gap comes from

DeepSeek opens with "the context does not contain the answer" in 15 of 100 (A) and 10 of 99 (B)
corpus answers, llama in 2 and 3. On those rows the relevance gap is -5.6 and -6.4; on the other 85
and 89 rows it is -0.28 and -0.24, inside the difference between repeats, with completeness -0.12
and -0.07 and faithfulness within 0.07. The largest gaps sit on paraphrases read literally ("a group
of processes ready to work" for a thread pool): llama fills in what the paraphrase means, DeepSeek
says the context has nothing on it. Relevance correlates with length at -0.06 and -0.08 for llama and
at +0.33 and +0.37 for DeepSeek only because the refusing answers are short. This split is read by
answer, as an explanation of the gap rather than a separate measurement.

## Outside the corpus

Counts per 100 questions, refusals read with rule 2
([the refusal entry](2026-09-13_a-refusal-the-rule-could-not-read.md)).

| pool | llama, A / B | DeepSeek, A / B |
|---|---|---|
| out of corpus, refused | 12 / 16 | 59 / 59 |
| off domain, refused | 57 / 58 | 93 / 86 |
| off domain, refused by the model among 62 that reached it | 19 / 20 | 55 / 48 |

38 of the 100 off-domain questions never reach a generator: retrieval returns nothing and the stand
answers "No relevant documents found." on its own. They are the same 38 questions in both arms and
both repeats, so they are counted in both arms' refusals and say nothing about either model; the last
row of the table removes them.

Where DeepSeek answers outside the corpus, the judge reads it as grounded and not relevant, and
llama the other way round: on `out_of_corpus` DeepSeek's faithfulness is +3.57 [+2.16, +4.96] and
+3.31 [+1.96, +4.67] higher and its relevance -2.86 [-3.87, -1.86] and -2.59 [-3.46, -1.77] lower
(n=70 in each repeat). DeepSeek says the context does not have it; llama answers from its own
memory, relevantly and without the context behind it.

## Also measured

- Language of the answer, on rows that answered: llama 229 of 231 and 226 of 226, DeepSeek 145 of 146
  and 153 of 153.
- Cut by the output ceiling: llama 2 and 1 of 262 calls, DeepSeek none.
- Latency on the corpus pool, p50: DeepSeek 4.9 and 4.5 s through the broker, llama 8.3 and 7.5 s on
  the card.
- Price: an arm of DeepSeek is 350 thousand input and 45 thousand output tokens (1167 and 150 per
  question), $0.0043; the whole cloud queue cost $0.0105.

## Against the predictions

| # | predicted | result |
|---|---|---|
| 1 | cloud floor: at most 30% equal answers, verdicts move at most 0.8 per axis, no 429 or 5xx | 0% equal and no errors held; verdicts did not (1.50 and 0.91) |
| 2 | corpus: DeepSeek above llama by 0.3-0.8 on faithfulness, 0.2-0.6 on relevance, 0.3-0.8 on completeness | missed: level on faithfulness, below on the other two |
| 2b | the models differ more than the repeats on at least one axis | held, on relevance |
| 3 | repeats of one model move verdicts at most 0.5 per axis | missed, about a point |
| 4 | answer language: DeepSeek 100%, llama at least 95% | llama held; DeepSeek one row short in A |
| 5 | off-domain refusals within 10 points of each other | missed: 28 to 36 points apart, and 55 against 19 and 48 against 20 of the 62 questions that reached a model |
| 6 | cuts by length: llama at most 2%, DeepSeek none | held |
| 7 | about 1.5 million input tokens per DeepSeek arm, under $0.05 in all | the price held; the tokens were four times fewer |

## Decision

- Nothing changes by default. The entry answers the owner's question, it does not propose a new
  generator: on the corpus the two are level on faithfulness and the 8b is ahead on relevance through
  answering where DeepSeek declines.
- Whether declining on a free paraphrase is the better behaviour is the owner's call, not the judge's.
- Found while writing this entry: a row the stand refuses before any generator call is stamped with
  the role's default generator, so a reading by the stamped model would credit the stand's refusal to
  a model. It is recorded in the arc's log for a fix.

## Caveats

- One judge, a 7B Qwen. Whether it prefers llama's style was checked with DeepSeek as the judge
  (experiments 61-63); that check has no entry yet.
- `single_shot` only. The stand's default path is the agent, where a question costs 2.3 times the
  tokens; the cloud arms were not run through it.
- The corpus pool is 100 Russian paraphrases; every pool is 100 questions per arm.
- The cloud's floor was taken on 50 other questions and under the older JSON rule.
- The broker decides the cloud model's weights, precision and node, and they may change between
  calls; the floor pair was measured in one window.
