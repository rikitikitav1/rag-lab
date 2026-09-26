# 2026-09-09 - Same rows judged on two engines: 39% of verdicts differ, not attributable to the engine

Arc 5 puts a second engine on the stand, so the first question a reader will ask of any two-engine
result is whether the engine moved the verdicts. This entry measures the move on identical rows, and
then explains why the number cannot be charged to the engine.

## Setup

**Set** `paraphrased_ru` and `out_of_corpus` (n=10 rows, 28 scored verdicts across three axes) ·
**corpus** `clean_1024`, `k` 5, agent path, `max_hops` 4 · **source run** `arc5_vllm_judge_smoke`,
answered 06.09 by `llama3.1:8b`, copied unjudged into each arm by `POST /v1/eval/rejudge`, so both
arms judge the same answers over the same contexts · **judge prompts** `judge_faithfulness` 2,
`judge_relevance` 2, `judge_completeness` 2, the same versions in both arms · **temperature** 0,
`seed` 0, one row at a time · **arm A** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM
`0.29.1.dev1+g98dff2a81`, batch invariance on, 493.9 s · **arm D** `qwen2.5:7b` Q4_K_M on ollama
0.32.0, `num_ctx` 8192, resident on the card, 66.2 s

## Result

Scores per question, faithfulness / relevance / completeness:

```
question   vLLM AWQ      ollama Q4_K_M
10617      8/10/7        7/9/7
10657      7/10/7        9/10/7
10685      7/10/10       7/7/10
10727      9/10/10       7/10/10
10741      10/10/7       10/10/7
10749      9/10/7        7/10/7
10759      9/10/0        7/9/0
10769      7/8/7         7/7/7
74644      7/10/-        7/10/-
74649      4/4/-         0/1/-
```

**11 of 28 verdicts differ, 39%.** For scale, the same judge on ollama across a reload moves 14% of
its scores on byte-identical input (arc 4, n=50), and the same judge on vLLM across a reload moved 0
of 28 in the companion entry. A point estimate on ten rows; no interval is claimed.

The largest single divergence is question 74649, an out-of-corpus row: 4/4 against 0/1. Both arms
agree the answer is poor and disagree on how poor, which is the shape a reader should expect from a
judge whose scale is not anchored.

## What this cannot say

It cannot say the engine moved the verdicts. The two arms differ in **two** things: the engine and
the quantisation. vLLM cannot read the GGUF that ollama serves (measured: this build's
`QUANTIZATION_METHODS` lists AWQ, GPTQ, FP8 and others, and GGUF is not among them), and an
unquantised 7B does not fit in 8188 MiB. So on this card the pair cannot be separated, and 39% is
the divergence of the pair (vLLM, AWQ) against (ollama, Q4_K_M).

This is the arc's own opening rule applied to the arc's own data: a hand that changes engine and
quantisation at once is a descriptive pair, not a test of a factor.

The judge prompt sizes were also recorded, because ollama trims whole messages to fit its window and
reports the trimmed count without complaining, which would make it both faster and worse in a way
nothing in the record would show. The largest judge prompt over these rows is 1959 tokens against a
window of 8192. Nothing was trimmed, and the field now says so for every future run.

## Decision

The disqualification already in `compare` stands and now has a number behind it rather than an
argument: arms judged on differently named engines are not comparable. Nothing else was decided
here; the 39% is not a cost of switching engines, and must not be quoted as one.

## Caveats

- Ten rows. The full set is needed before this is a number rather than a smoke.
- Engine and quantisation are confounded, as above. Separating them needs weights both engines can
  serve at the same precision, which on this card means a small model in fp16 rather than a 7B.
- Arm D ran on the card, arm A on the card, but the vLLM stamp cannot say so: `on_card` asks
  ollama's `/api/ps`, which does not exist on vLLM, and the swallowed error is recorded as "unknown"
  rather than "not applicable". Open with the auditor.
- The chat templates are not identical either. Measured the same day: an empty `system` message
  costs 11 tokens on ollama and 16 on vLLM, because ollama drops the block entirely and vLLM emits
  an empty one. This run sends no empty system, so it is not implicated here, but it is a second
  difference living inside "the engine".
