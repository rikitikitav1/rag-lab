# 2026-07-29 - Judge vs judge: qwen2.5 7b against 32b on the same 100 answers (Spearman)

Question: how much of our metric is the ruler, not the thing measured? Cheap surrogate for human
calibration (audit item): snapshot the 7b judge's verdicts on `ss_70b_k5`, null the axes, re-judge
the identical answers with qwen2.5:32b, correlate.

| axis | Spearman | mean 7b -> 32b | exact match |
|------|----------|----------------|-------------|
| faithfulness | 0.30 | 7.60 -> 9.35 | 26% |
| relevance | 0.52 | 8.51 -> 9.23 | 59% |
| completeness | **0.71** | 6.09 -> 5.85 | 60% |

Reading:
- **Completeness is the most judge-stable axis** - the only one with an anchor (reference answer).
  Rankings and means both survive the judge swap.
- **Faithfulness is the most judge-dependent.** The 32b judge pushes almost everything to 9-10
  (ceiling effect); with the scores clumped, ranks collapse into ties and rank correlation has
  nothing to grab. 7b was harsher and more discriminating - a bigger judge is not automatically
  a better ruler.
- Practical rule recorded: **absolute axis values do not transfer between judges; A/B comparisons
  under one judge remain valid.** "We changed the judge and faithfulness jumped +1.75 out of thin
  air" is the cleanest illustration yet of why metrics only compare within one procedure.

## Follow-up (overnight): second run + the size verdict replicated across judges

The 32b judge finished re-judging `ss_8b_k5` as well (15.4h on CPU). Two additions.

Spearman 7b vs 32b on the second run (same procedure):

| axis | Spearman (ss_8b_k5) | vs ss_70b_k5 | mean 7b -> 32b |
|------|---------------------|--------------|----------------|
| faithfulness | 0.47 | 0.30 | 7.22 -> 8.61 |
| relevance | 0.62 | 0.52 | 8.55 -> 9.10 |
| completeness | 0.59 | 0.71 | 6.19 -> 5.94 |

Same shape: moderate rank agreement, 32b generous on faithfulness/relevance, close on
completeness. Faithfulness agrees better here than on the 70b run (0.47 vs 0.30) - the 32b
judge scores 8b answers lower on average (8.61 vs 9.35), so less ceiling, less tie collapse.
Consistent with "0.30 was the ceiling artifact, not judge disagreement".

The bonus question - does "70b is more grounded" survive a judge swap? Paired deltas
(70b - 8b) per question under the 32b judge:

| axis | mean delta | Wilcoxon p | verdict under 7b (before) |
|------|-----------|------------|---------------------------|
| faithfulness | **+0.74** | **9e-05** | +0.38, p=0.034 (borderline) |
| relevance | +0.13 | 0.62 | ~0, n.s. |
| completeness | -0.09 | 0.57 | ~0, n.s. |

The verdict not only holds - it strengthens: the stronger judge sees a bigger faithfulness
gap, and both judges independently agree on the direction and on where there is no
difference. Cross-judge replication is the strongest confirmation available without a human
panel. The deployment decision (stay on 8b, 23x latency) is unchanged; the finding
"a bigger generator is measurably more grounded" is now robust.
