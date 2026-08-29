# 2026-07-29 - Judge vs judge: qwen2.5 7b against 32b on the same 100 answers (Spearman)

How much of the metric is the ruler rather than the thing measured? A human panel is out of reach,
so the cheap surrogate is a second judge: snapshot the 7b judge's verdicts, null the axes, re-judge
the identical answers with `qwen2.5:32b`, correlate.

## Setup

**Set** `paraphrased_ru` (n=100, run `ss_70b_k5`) · **corpus** pre-variant era, post-ablation · **judges** `qwen2.5:7b` against `qwen2.5:32b`, numeric 0-10

No regeneration: the answers are fixed and only the judge changes.

## Result

| axis | Spearman | mean 7b → 32b | exact match |
|------|----------|----------------|-------------|
| faithfulness | 0.30 | 7.60 → 9.35 | 26% |
| relevance | 0.52 | 8.51 → 9.23 | 59% |
| completeness | **0.71** | 6.09 → 5.85 | 60% |

Completeness is the most judge-stable axis, and it is the only one with an anchor, the reference
answer: rankings and means both survive the swap. Faithfulness is the most judge-dependent: the 32b
pushes almost everything to 9 and 10, and with the scores clumped the ranks collapse into ties, so
rank correlation has nothing to grab. The 7b was harsher and more discriminating, which is worth
recording plainly: a bigger judge is not automatically a better ruler.

## Follow-up: second run, and the size verdict replicated across judges

The 32b finished re-judging `ss_8b_k5` as well (15.4h on CPU), same procedure, n=100.

| axis | Spearman (ss_8b_k5) | vs ss_70b_k5 | mean 7b → 32b |
|------|---------------------|--------------|----------------|
| faithfulness | 0.47 | 0.30 | 7.22 → 8.61 |
| relevance | 0.62 | 0.52 | 8.55 → 9.10 |
| completeness | 0.59 | 0.71 | 6.19 → 5.94 |

Same shape: moderate rank agreement, the 32b generous on faithfulness and relevance, close on
completeness. Faithfulness agrees better here than on the 70b run (0.47 against 0.30) because the
32b scores 8b answers lower on average, so there is less ceiling and less tie collapse. That is
consistent with 0.30 being a ceiling artifact rather than judge disagreement.

Does "the 70b is more grounded" survive a judge swap? Paired deltas (70b minus 8b) per question
under the 32b judge:

| axis | mean delta | Wilcoxon p | verdict under 7b |
|------|-----------|------------|---------------------------|
| faithfulness | **+0.74** | **9e-05** | +0.38, p=0.034 (borderline) |
| relevance | +0.13 | 0.62 | about 0, n.s. |
| completeness | -0.09 | 0.57 | about 0, n.s. |

The verdict strengthens: the stronger judge sees a bigger gap, and both judges agree on the
direction and on where there is no difference.

## Decision

Recorded as a rule: absolute axis values do not transfer between judges, A/B comparisons under one
judge remain valid. "We changed the judge and faithfulness jumped +1.75" is the cleanest
illustration of why metrics only compare within one procedure. The deployment decision is
unchanged (stay on 8b, 23x latency); the finding that a bigger generator is measurably more
grounded now has cross-judge replication behind it.

## Caveats

- two judges from one family, so shared training biases would not show up as disagreement
- exact-match percentages depend on the 0-10 grid; a coarser or finer rubric would move them
  without anything else changing
- the 32b ran on CPU, which is why this is a one-off calibration and not a routine second opinion
