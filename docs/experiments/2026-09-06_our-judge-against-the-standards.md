# 2026-09-06 - Our judge against the standard's, and the ruler that had to come first

The second sub-phase of the standard arc asks whether the faithfulness this stand has been
reporting for two months measures the same thing RAGAS calls faithfulness. Unlike the retrieval
half, both sides here call a model, so the comparison has a floor: two readings of the same row
disagree even when nothing changed. That floor was measured last, and it turned out to sit above
one of the results this entry set out to report.

## Setup

**Rows** the corpus pool of `arc3_agent_baseline` alone, answered by our own outcome, carrying a
non-empty context: n=93 · **corpus** `clean_1024`, unchanged between the two readings, this compares
two judges of one set of answers, not two corpora · **judge** ours is `qwen2.5:7b` at role `judging`;
the guest is the same model behind the same client

- our side: `metrics.faithfulness`, 0-10 integer, `evals/generation_metrics` schema 2
- the guest's side: `Faithfulness` from ragas 0.4.3, 0-1, driven by `evals/guest_llm.OurClient`,
  a `BaseRagasLLM` over our own `llm.ask`, so both sides share the client, the seed and the
  card guard
- the population is the corpus pool only. Three pools sit at three heights, and a first reading
  that mixed them reported a correlation the gap between pools had propped up
- refusals are excluded by our own outcome, not by the guest's `nan`
- numbers from `datasets/measurements/judge_correlation_20260906.json` (schema 3), reproduced by
  `scripts/ragas_judge_correlation.py`; the floor from `guest_noise_floor_20260906.json`

## What was declared before the run

Four predictions with their refutation conditions, written into the arc log before the count:

1. the guest's correlation with a deterministic overlap covariate stays below 0.3, else the axis
   is reading lexical overlap rather than groundedness
2. the partial correlation, controlling for that covariate, differs from the plain one by less
   than 0.1, else the agreement is carried by overlap
3. the two strata by code share of the context differ by less than 0.1
4. n is at least 100

A fifth rule was written after the fact and applies from here on: a prediction whose interval
covers its threshold is **unchecked at this n**, neither confirmed nor refuted.

## Result

```
rho, ours against the guest      0.5001   [0.3641, 0.6127]   n=93
rho, guest against overlap       0.2235        below 0.3, prediction 1 holds
partial, controlling overlap     0.4701        0.03 from the plain, prediction 2 holds
strata by code share             gap 0.048     prediction 3 holds
n                                93            prediction 4 fails
```

Three controls of four hold and the fourth fails, so by the rule the band on rho is not read.
The record says: the two judges are related, and how strongly is unclear at this n.

**The number that was withdrawn.** A first reading gave 0.627 at n=137. It was computed over
three pools at once, and the height difference between pools was doing part of the work. On the
declared population the same code gives 0.500. The 0.627 is not a second measurement of the same
thing; it is the same measurement made wrong.

## The ruler, measured after everything else

The same 300 rows were copied and scored by the guests a second time, one residency apart, nothing
else changed. The pass was stopped by the owner with two dozen rows still owed, because the guest
had failed to parse them three times running.

```
pairs 321, identical 225 (70%), moved 96 (30%)
mean absolute delta   0.0719
mean signed delta     0.0093  [-0.0081, +0.0279]   no drift
```

So the guest axes move by about **0.07** when nothing changed. Our own judge, checked the same day
on five known cases three times each, gave the same answer every time. The non-determinism arrives
from inside `ragas`, not from the client the two sides share.

Two caveats belong beside this number. The pass covers only rows the guest could parse **twice**:
19 of 886 calls failed to parse, concentrated on Russian answers and on long statement lists where
the model returns a dict where the library expects a string. And the floor is a mean over three
axes that differ: 0.079 for faithfulness, 0.039 for context precision, 0.099 for context recall.

## What the ruler did to the rest of the sub-phase

Three side measurements were taken before the floor existed. Two survive it and one does not.

```
hypothesis "the axis reads overlap, not grounding", two arms:
  keep the meaning, remove the overlap     copy 1.000 -> paraphrase 0.982    delta 0.018
  keep the overlap, break the meaning      copy 1.000 -> negated    0.448    delta 0.552
"our judge is stricter on code", paired      kept 0.826 vs stripped 0.846    delta -0.020
```

The negated arm is nearly eight times the floor and stands: an answer that keeps every word of the
context but reverses its claim loses half the score, so the axis is not reading overlap. The
paraphrase arm is **below the floor** and can no longer be read as a measured effect; what it
showed is that the axis does not collapse when overlap goes, which is weaker than what was written
at the time. The code premium was already reported as not found, and the floor explains why that
measurement could never have found it: the effect under dispute was 0.1, the instrument moves 0.07,
and the interval was [-0.162, +0.095].

## What the move to the standard cost

Measured on a ten-row smoke through the real judging handler: our three axes cost 69.2 s, the
guests' three cost 595.1 s, a multiplier of **9.6**. A second smoke with the library's telemetry
switched off gave 29.2 s per row for faithfulness against 26.9 s with it, that is no effect visible
over the difference between two ten-row sets. The multiplier is a measurement, not an upper bound
inflated by a stray network call, and the guests became a job of their own that no run waits on.

## Reading

- the two judges agree enough to be talking about the same property, and the strength of that
  agreement is not established at n=93
- effects smaller than 0.07 on the guest axes are inside the instrument. Three of this sub-phase's
  own numbers sit there, and only the negated arm is clear of it
- the guest is the noisy half of the pair. Where a design can choose, our own axis is the cheaper
  and the steadier one, and the guest earns its 9.6x only where it measures something ours does not
- one thing the two do not share: on a refusal our judge scores by "no unsupported claims" and the
  guest by "how many claims the context entails", so they diverge by construction there. That is
  why refusals are outside the population, and the decision to exclude them now has a reason
  beside itself
