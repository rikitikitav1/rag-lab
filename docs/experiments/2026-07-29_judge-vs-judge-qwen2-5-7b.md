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
