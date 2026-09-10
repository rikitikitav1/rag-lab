# 2026-09-09 - What batch invariance costs on an AWQ judge, and what it buys

vLLM was brought onto this stand for one reason: its `VLLM_BATCH_INVARIANT` flag promises a judge
that returns the same verdict on the same input, where ollama moves 14% of its scores across a
reload. The flag was believed to be worth its cost without the cost ever being measured on the axis
that matters. This entry measures it, and finds the flag is not what buys the zero.

## Setup

**Set** `paraphrased_ru` and `out_of_corpus` (n=10 rows, 28 scored verdicts across three axes) ·
**corpus** `clean_1024`, `k` 5, agent path, `max_hops` 4 · **source run** `arc5_vllm_judge_smoke`,
answered 06.09 by `llama3.1:8b`, copied unjudged into each pass by `POST /v1/eval/rejudge` so the
material is identical in every pass · **judge** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM
`0.29.1.dev1+g98dff2a81`, prompts `judge_faithfulness` 2, `judge_relevance` 2,
`judge_completeness` 2, `temperature` 0, `seed` 0, one row at a time (`judge_width` 1) ·
**vLLM start** `--enforce-eager --max-num-seqs 16 --max-model-len 8192`,
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, `--gpu-memory-utilization 0.95` on an RTX 4070 Laptop
of 8188 MiB · **ollama arm** `qwen2.5:7b` Q4_K_M, `num_ctx` 8192, resident on the card

Only one thing varies between passes A, C and E: the environment variable `VLLM_BATCH_INVARIANT`.
The script takes it as `INV_OVERRIDE` so the two configurations are one substitution rather than two
code paths that can drift apart. Which value a pass actually ran under is read back from the server
into every judged row as `engine_added.batch_invariant`, not from memory of what was typed.

## Result

| pass | engine | flag | reloaded before | elapsed | verdicts |
|---|---|---|---|---|---|
| A | vLLM AWQ | on | yes | 493.9 s | baseline |
| B | vLLM AWQ | on | yes | 512.7 s | identical to A |
| C | vLLM AWQ | off | yes | 56.7 s | identical to A |
| E | vLLM AWQ | off | yes | 57.2 s | identical to A |
| D | ollama Q4_K_M | none | no | 66.2 s | 11 of 28 differ, see the linked entry |

All 28 verdicts of A, B, C and E agree exactly. This is a point estimate on ten rows, not an
interval: the run is small enough that the honest statement is "no verdict moved", not "the floor
is zero within some bound".

## Why the flag buys nothing here

Batch invariance fixes non-determinism that arises when the composition of a batch changes: a row
lands beside different neighbours on a second run, the reduction happens in a different order, and
the last bits differ. This stand judges one row at a time. `OLLAMA_NUM_PARALLEL` is pinned to 1 and
`judge_width` reads it, so every batch holds exactly one sequence and there is nothing to reorder.

Pass E is the cell that makes this a finding rather than a guess: with the flag **off**, across an
engine reload, the verdicts still did not move. The zero comes from the engine and the workload, not
from the flag.

The flag becomes load bearing the moment judging runs several rows at once. Nothing on this stand
does that today.

## What the 8.7x is, and what it is not

8.7x is the cost of the flag **on an AWQ model**. Under the flag vLLM replaces fused kernels with
unfused reference ones, and for a weight-only-quantised model that includes dequantising the weights
and running a plain matmul instead of the AWQ kernel. On an unquantised model the same flag would
cost something else. The number belongs to the pair (flag, quantisation), and quoting it as "the
price of batch invariance" would repeat the error this arc keeps catching.

What was withdrawn on the way here: an earlier reading of a run with the flag off reported a 6992
window and blamed the flag, when that run differed in three parameters at once. A later claim that
"the judge on vLLM is seven times slower" blamed the engine for what is now measured to be the flag.
Both came from the same move, several differences and one culprit.

## Decision

Not taken here. Turning the flag off for single-stream judging is the owner's call, and the case for
it is above: on this workload it costs 8.7x and changes no verdict. Recorded as a proposal, not as a
new default; the arc's checklist carries the open item to decompose the 8.7x itself.

## Caveats

- Ten rows, 28 verdicts. A floor claim needs the full set before it is a number rather than a smoke.
- Only single-stream judging was measured. The flag's whole purpose lies outside that regime.
- The AWQ quantisation is held constant across A, C and E, so it does not confound the comparison
  between them, but it does colour the magnitude, as above.
- Pass D sits in this table for scale only. It changes engine and quantisation together, and what
  that comparison can and cannot say has its own entry.
- The three vLLM passes ran with `VLLM_SERVER_DEV_MODE=1` so the flag's value could be read back
  from `/server_info`. Without it the field is absent from the record, which reads as "nowhere to
  ask", not as "the flag was off".

## Update, measured on 2026-09-10: the same question at n=300

The entry above closed with a caveat: ten rows and 28 verdicts are a smoke, not a floor. This is the
same question asked of a real set.

**Set** `paraphrased_ru` (n=300 rows, 144 of them scorable, 388 verdicts) · **corpus** `clean_1024`,
agent path · **source run** `arc4_lang_after`, answered 08.09, copied unjudged into each pass by
`POST /v1/eval/rejudge` · **judge** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM
`0.29.1.dev1+g98dff2a81`, batch invariance **off**, `judge_width` 1, prompts `judge_faithfulness` 2,
`judge_relevance` 2, `judge_completeness` 2 · the engine container is destroyed and started again
between every pass, so each pass judges in its own residency

| pass | job | elapsed | rows scored | verdicts | residency |
|---|---|---|---|---|---|
| A | 2539 | 14 min | 144 of 300 | 388 | 2539 |
| B | 2540 | 14 min | 144 | 388 | 2540 |
| C | 2541 | 14 min | 144 | 388 | 2541 |

**Nothing moved.** Not one score and not one reason text, across A against B and A against C: 776
score comparisons and 776 free-text comparisons, all identical. The same judge on ollama moves 14% of
its scores and 58% of its reason texts across a reload.

The flag was off for all three passes, which is the point: at `judge_width` 1 the batch never varies,
so there is nothing for batch invariance to fix, and the 8.7x it costs buys nothing. That claim now
rests on 388 verdicts rather than 28.

**Two things this does not say.** It says nothing about judging several rows at once, which is the
regime the flag exists for. And it is one model on one engine: a floor belongs to a pair, so ollama's
14% and this zero are not two readings of one instrument.

**A prediction that missed, recorded because the reason is instructive.** Before the run the entry's
author predicted 25 to 35 minutes per pass, from the smoke's 5.67 seconds per row times 300. The
passes took 14 minutes. The rate was right and the unit was wrong: eight of the smoke's ten rows owed
all three axes, while only 144 of the 300 do, so the work was 388 verdicts rather than 900. Per
verdict the smoke said 2.02 seconds and the run gave 2.16, a 7% difference.

**Also confirmed at scale**: the largest judge prompt over the three passes was 4398 tokens against a
window of 8192, so ollama's silent trimming has no room to happen on this material either.
