# 2026-07-28 - k-sweep re-judged with a numeric 0-10 judge (the categorical verdict reversed)

The categorical judge buries a good-but-lightly-extrapolated answer and a mostly-invented one in
the same "partially" bucket. If the metric hides the difference, the k-curve it produced cannot be
trusted, so the same runs are re-scored with a numeric judge. This is a change to the ruler, not to
the system.

## Setup

**Set** `paraphrased_ru` (n=100) · **corpus** pre-variant era, post-ablation · **judge** `qwen2.5:7b`, numeric 0-10 (`judge_*.v2`)

Numeric 0-10 judge: rubric per axis, `score` through structured output, `judge.py` and
`generation_metrics.py` on averages, prompts `judge_*.v2`. The same runs are re-judged with no
regeneration. Agent pipeline, temp 0.

## Result

| k | faithfulness | relevance | completeness |
|---|-----:|-----:|-----:|
| 1 | 6.79 | 8.60 | 5.48 |
| 3 | 6.98 | 8.68 | 5.86 |
| **5** | **7.18** | 8.90 | **6.16** |
| 6 | 6.83 | 8.87 | 5.99 |
| 7 | 6.78 | **9.06** | 6.06 |
| 10 | 6.99 | 8.94 | 6.09 |

The numeric judge reversed the categorical verdict. Categorical said faithfulness peaks at k=3 (41)
and collapses to 22, that is, wider retrieval hurts grounding. Numeric shows faithfulness rising to
a peak at k=5 (7.18) and then plateauing between 6.8 and 7.2. The mechanism is visible in the two
scales: as k grows, faithful answers drift into "partially", so the counter drops while the average
does not, because those answers are sevens. The categorical metric was wrong in the sign of the
conclusion, not only in its size. Peaks: faithfulness and completeness at k=5, relevance at k=7.

## What a composite does with this curve

RRF over the three axes (k=60, equal weights) puts k=5 first (0.0487): first on faithfulness and
completeness, third on relevance. k=10 is second and stable on every axis; k=7 is third, first on
relevance and last on faithfulness, which sinks it. Fusion rewards stability over a single peak.

Moving the optimum to k=7 by weighting takes a 5x weight on relevance, which is the same as
dropping the other two axes; at 3x, k=10 wins on stability. If one axis dominates that hard, the
composite is the wrong tool and argmax on that axis is the honest answer.

The better framing is constrained rather than weighted. Faithfulness is a floor, since a fluent
ungrounded answer is the worst case a RAG system can produce, and relevance is the objective:
maximise relevance subject to faithfulness above a threshold. Faithfulness plateaus above k=1, so
the threshold admits k=3 through k=10 and the best relevance among them is k=7. Where the floor
sits is a product decision rather than a measurement.

## Decision

Default k=5 (`results_limit` 3 → 5): the composite optimum, best faithfulness and completeness,
relevance 0.16 below its own peak.

## Correction, measured the same day

The paired statistics in
[the significance entry](2026-07-28_paired-significance-testing-lands-in-the.md) show k=5 against
k=10 to be indistinguishable on every axis (faithfulness p=0.37). The k=5 default stands on cost
rather than on proven quality: at equal quality, feed the generator half the chunks.

## Caveats

- one run per k at n=100, and this entry reports point estimates only; the intervals arrived a day
  later and changed what the numbers support
- the re-judge shares its answers with the categorical run, so the two eras differ in the ruler
  alone, which is the only reason the comparison above is legitimate
