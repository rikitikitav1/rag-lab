# 2026-08-29 - Generator grid: 4b against 8b, reranking, and two languages

How much of an answer is the generator, when the facts are brought by retrieval? Eight arms
crossing model, reranking and language, plus a ninth arm that repeats the first one a day later to
measure what the stand produces out of nothing. The ninth arm is why half of this entry's numbers
are labelled unreadable: the judge moved on its own, and it moved further than the effects.

## Setup

**Set** `paraphrased_v2_ru` (n=823) and `paraphrased_v2` (n=820) · **corpus** `clean_1024` ·
**pipeline** `single_shot`, phased · **judge** `qwen2.5:7b`, prompts v2, temperature 0

```
model    gemma3:4b | llama3.1:8b
rerank   off | on
language ru | en
control  arm 1 repeated at the end, same settings
```

Within a cell the two models see the same context by construction: `_phase_retrieve` does not know
the generator, depth resolves once into the snapshot, and hnsw at a fixed depth is deterministic.
The control arm confirms this rather than assuming it: context and sources matched byte for byte on
823 of 823.

The decision rule was written before the run, in `notes/arc/hygiene_arc_log.md`, sections
"Предрегистрация: модель генерации 4b против 8b" and "Поправка к предрегистрации". Six predictions,
each with its falsification. Half-widths 0.088 / 0.140 / 0.117 were computed before the run from a
control pair of 100 questions.

Code: `app/evals/runner.py`, `app/use_cases/retrieval_compare.py`, `app/use_cases/judge.py`.
Arms placed by a sequencer, one at a time, each judged before the next started.

## Result

Arm means, judge scores 0-10.

| arm | n | faithfulness | relevance | completeness | tokens |
|---|---|---|---|---|---|
| gemma ru plain | 823 | 7.54 | 7.47 | 5.79 | 259 |
| gemma ru rerank | 823 | 7.79 | 7.79 | 6.07 | 257 |
| llama ru plain | 823 | 7.39 | 8.66 | 6.17 | 305 |
| llama ru rerank | 823 | 7.36 | 8.81 | 6.36 | 307 |
| gemma en plain | 820 | 8.31 | 7.39 | 5.86 | 271 |
| gemma en rerank | 820 | 8.23 | 7.50 | 5.94 | 275 |
| llama en plain | 820 | 8.33 | 9.06 | 6.36 | 194 |
| llama en rerank | 820 | 8.16 | 9.04 | 6.35 | 198 |

Model effect, llama minus gemma, paired, bootstrap interval over eight seeds. A bound that changes
sign across seeds is reported as parity rather than as a result.

| cell | faithfulness | relevance | completeness |
|---|---|---|---|
| ru plain | -0.1458 **unreadable** | +1.1908 [+0.9830, +1.3876] | +0.3791 [+0.2309, +0.5237] |
| ru rerank | -0.4301 **unreadable** | +1.0194 [+0.8153, +1.2053] | +0.2855 [+0.1458, +0.4204] |
| en plain | +0.0171 **unreadable** | +1.6646 [+1.4744, +1.8537] | +0.4915 [+0.3415, +0.6439] |
| en rerank | -0.0707 **unreadable** | +1.5341 [+1.3500, +1.7110] | +0.4171 [+0.2793, +0.5561] |

Rerank effect, rerank minus plain, paired, same set and corpus as the row above.

| cell | faithfulness | relevance | completeness |
|---|---|---|---|
| ru gemma | +0.2479 **unreadable** | +0.3244 [+0.1106, +0.5298] | +0.2770 [+0.1106, +0.4326] |
| ru llama | -0.0365 **unreadable** | +0.1531 [-0.0219, +0.3269] | +0.1835 [+0.0328, +0.3220] |
| en gemma | -0.0817 **unreadable** | +0.1061 [-0.1012, +0.3159] | +0.0744 [-0.0854, +0.2366] |
| en llama | -0.1695 **unreadable** | -0.0244 [-0.1598, +0.1049] | 0.0000 [-0.1366, +0.1293] |

`unreadable` is not a hedge, it is the control arm's verdict, below.

## The control arm, and why faithfulness does not survive it

The ninth arm repeats arm 1 with the same settings a day later. Prediction: the difference stays
inside the half-widths. Paired, `paraphrased_v2_ru`, n=823, `clean_1024`:

| metric | drift | half-width | verdict |
|---|---|---|---|
| faithfulness | +0.2442 [+0.1434, +0.3572] | 0.088 | **outside, by 2.8x** |
| relevance | +0.0097 [-0.1434, +0.1519] | 0.140 | inside |
| completeness | +0.0316 [-0.0693, +0.1300] | 0.117 | inside |

The mechanism is the judge, not the generator, and it is measured rather than argued:

```
context identical byte for byte   823 of 823
sources identical                 823 of 823
answer identical byte for byte     52 of 823   (generation runs at temperature 0.1)

on those 52 identical inputs:  faithfulness +0.4038,  43 unchanged, 9 up, 0 down
```

Same context, same answer, same prompt version, judge temperature 0, and nine scores went up and
none went down. Across all 823 pairs: 553 scores unchanged, 174 up, 96 down, mean absolute move
0.823. Ruled out as causes: prompt version (all three judge prompts v2 active since 27.07), the
ollama process (up since 28.08 07:06 UTC, never restarted, window 8192), judge temperature, worker
code (the tree was frozen for the duration). The one known difference: when arm 1 was judged the
card held only the judge, and when arm 9 was judged a resident reranker sat beside it (~1.1 GB of
torch in the worker process, left over from arm 8).

Only faithfulness moved. Relevance is judged against the question, completeness against the
reference, and faithfulness is the only one that reads the context.

Consequently every faithfulness delta in this entry reads as **direction plausible, magnitude not
measurable**, until the judge is calibrated. Relevance and completeness stand: drift of 0.010 and
0.032 against effects of 0.28 to 1.66.

## The half-width was the wrong yardstick, and that is a mistake in the pre-registration

The half-widths were derived from the sd of paired deltas of a 100-question control pair. That is
the **resolution of the set**: the difference the set can tell apart. It says nothing about the
**systematic offset of the stand**: the difference the stand manufactures from nothing. The first
does not bound the second, and comparing a control arm against a half-width was wrong from the
start. The right threshold for a control arm is its own interval, and [+0.1434, +0.3572] does not
cover zero.

## Predictions

| # | prediction | outcome |
|---|---|---|
| 1 | llama not below gemma on faithfulness in 3 of 4 cells | **not evaluable**, judge not calibrated |
| 2 | no cell reaches abs 0.5 on faithfulness (amended from 1.0) | **not evaluable** |
| 3 | completeness gap exceeds faithfulness gap in 3 of 4 cells | holds, 3 of 4 |
| 4 | gemma's language discipline at least 95% on both sides | **falsified**: 0.933 raw, 0.947 counting prose only, against 1.000 on the English arms |
| 5 | on Russian the rerank effect on faithfulness is smaller than the model effect | **not evaluable**, and under-specified: it never said which cell supplies the model effect |
| 6 | each model's faithfulness on English is not below its Russian | **not evaluable** |

Prediction 3 holds, but its stated reason ("completeness has more room below the judge's ceiling")
is no longer the only candidate: part of faithfulness's flatness is its own noise.

Prediction 4 is the one that decides a model on its own, by the pre-registration's own wording, and
it failed on the Russian side under both readings of the metric.

## Cost

| arm | wall clock |
|---|---|
| gemma ru plain / rerank | 66.3 / 68.6 min |
| llama ru plain / rerank | 116.1 / 118.9 min |
| gemma en plain / rerank | 68.0 / 71.6 min |
| llama en plain / rerank | 74.5 / 79.6 min |

25.6 hours of stand time in total, nine runs and nine judge passes, 7 395 answers each scored on
three axes. Reranking costs gemma 2.3 minutes per 823 questions, that is 168 ms per question and
3.4% of an arm, which sits between the 86 ms measured on an idle card and the 156 ms on a busy one.
The 2.76 s in `datasets/measurements/rerank_latency.json` is the CPU price and does not apply to a
phased run.

The pre-registration estimated 6.5 s per question for gemma and 10.1 for llama, about a day for
eight arms. Actual: 4.8 s and 8.5 s, arms of 66 to 80 minutes. The estimate was taken before the
pipeline became phased, and batched embedding plus batched reranking took nearly half of it back.
A cost measured before the pipeline is rebuilt prices the old pipeline.

## Decision

`gemma3:4b` becomes the generator and reranking becomes the default, both in one commit. The owner
named the pair and the condition on 28.08 ("if the ratios are roughly acceptable, gemma is our
generator") and confirmed it on 29.08. The deciding argument is headroom rather than scores: llama
at 5926 MiB plus the embedder at 851 plus the reranker's weights leaves nothing on an 8188 MiB card,
and `grade_documents`, `rewrite_question`, a query router and RAGAS all still have to fit. With
gemma at 4248 there is about 1.5 GB left.

The price is written down rather than rounded away: gemma with reranking against llama without
gives relevance **-0.866** on Russian and **-1.559** on English, completeness -0.102 (parity) and
-0.417, faithfulness +0.394 and +0.099 (both unreadable). Llama's advantage is not verbosity: on
English it writes 194 tokens against gemma's 271 and still wins relevance. That gap is a debt owed
by the next round of generator-prompt work, and it is pre-registered as that round's target.

Three things must change with the default, or the change does not take effect where it is meant to:
`RERANK_DEVICE: cpu` is hardcoded for the API in `docker-compose.yml`; preflight's residency checks
see only ollama models, so the reranker is invisible to them; and `resolve_rerank(None)` reads
`rerank.enabled`, so flipping it changes every run that did not ask.

Not decided here: whether reranking is worth its quality on English. The one arm that showed harm
(llama, faithfulness -0.1695) is unreadable for the same reason as the rest.

## Caveats

- **Every faithfulness number above is direction-only.** The stand moves 0.244 on that axis between
  two identical runs, which is larger than most of the effects. Relevance and completeness are not
  affected: their drift is 0.010 and 0.032.
- Two models at 4b and 8b are a narrow band. Nothing here generalises to "small models" or "large
  models".
- One judge, 7b, scores ordinal. A gap of 0.1 is not "10% better".
- The two sets are different sizes (823 and 820) and are not interchangeable. Numbers from one are
  not measurements of the other.
- `single_shot` only. The agent pipeline was not run.
- The reranker's ~1.1 GB is an estimate from weights, not a measurement; the 3149 MiB seen in the
  sampler is the batch mode, which is not what an interactive request does.
- Reading the same questions across languages is a separate entry, and it is post-hoc:
  [the same question in two languages](2026-08-29_the-same-question-in-two-languages.md).

## Correction, 30.08: the decision was reverted the next day

`gemma3:4b` was the generator for about twenty-four hours. It has **no tool calling in ollama at
all** (`ollama show`: `completion, vision`; no gemma3 tag has it, and the refusal is a 400 raised
before inference), so the agent pipeline answered `refused` with zero sources to every question:
400 of 400 across four control runs. The caveat above says it plainly, "`single_shot` only, the
agent pipeline was not run", and the decision was taken past that line anyway. The generation role
is one role for both pipelines, and only one of them was measured.

The owner reverted on 30.08: the generation role is `llama3.1:8b` again, and **reranking is off**,
because the 8b takes the card room the reranker held. Nothing measured here is withdrawn. The
ratios above still describe what the two models do on single-shot answers, and the reranker is
still worth +0.0454 [+0.0250, +0.0670] of section MRR. What is withdrawn is the conclusion that
headroom decides: the headroom argument counted the models it knew about and did not count the
capability the agent needs.

Two numbers in this entry are also corrected by later measurement: the reranker's weights are
1728 MiB on the card, not the ~1.1 GB estimated here, and `llama3.1:8b` plus the embedder plus the
reranker do not fit, which is what the revert had to pay for.
