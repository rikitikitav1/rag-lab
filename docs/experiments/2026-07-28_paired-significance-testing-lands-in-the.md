# 2026-07-28 - Paired significance testing lands in the aggregator (the audit answered)

**Why:** an external audit named the sharpest weakness of the whole lab: point estimates without uncertainty. The RRF winner k=5 beat k=10 by 0.0487 vs 0.0484 - third-decimal territory - and rerank deltas of +0.2..0.4 on a 0-10 judge scale could live entirely inside noise. Since every comparison here is paired (the same question answered in both configs), paired statistics are nearly free.

**Change:** the aggregator now computes, for every experiment, `composite.pairwise` - winner vs each other value, per generation axis: mean paired delta, bootstrap 95% CI (10k resamples, fixed seed) and a Wilcoxon signed-rank p-value (rank-based: right for ordinal judge scores with outliers; zero-delta series short-circuit to p=1). Exposed via `pairwise_stats(run_a, run_b)` and stored in experiment `results`. The tolerant numeric parser is gone too: after the full re-judge of the categorical era (1442 logs re-scored by the numeric judge), a non-numeric verdict in the DB is a bug and now fails loudly instead of being silently skipped.

**What the statistics did to our own conclusions:**

| comparison | axis | delta | CI95 | p | verdict |
|---|---|---|---|---|---|
| 70b vs 8b (single_shot k5) | faithfulness | +0.38 | [-0.04, 0.81] | 0.034 | borderline real |
| 70b vs 8b | relevance | -0.04 | [-0.68, 0.60] | 0.58 | noise |
| 70b vs 8b | completeness | -0.10 | [-0.50, 0.31] | 0.49 | noise |
| k=5 vs k=10 (agent baseline) | faithfulness | +0.19 | [-0.19, 0.57] | 0.37 | indistinguishable |
| k=5 vs k=10 | relevance | -0.04 | [-0.49, 0.40] | 0.98 | indistinguishable |
| k=5 vs k=10 | completeness | +0.07 | [-0.33, 0.48] | 0.89 | indistinguishable |

**Honest corrections this forces:**
- **k=5 vs k=10 is a statistical coin flip on every axis.** The k=5 default stands, but the honest justification is now cost, not proven quality: at indistinguishable quality, feed the generator half the chunks. The earlier "k=5 wins the composite" phrasing overstated what the data supports.
- **The 70B's faithfulness gain is real-ish but marginal**: Wilcoxon says significant (p=0.034), the bootstrap CI grazes zero - a borderline effect. "23x latency for a borderline single-axis gain" is an even stronger version of the stay-on-8b decision.
- Rerank per-k deltas (completeness +0.2..0.4) remain to be tested pair-wise; treat them as suggestive until then.

**Meta:** the lab's own headline ("fix the ruler before optimizing") now applies twice: first the judge's resolution (categorical → numeric), then the conclusion's resolution (point estimate → interval + significance). Both times the correction changed what we would have claimed.

**Postscript - the multiple comparisons trap (found immediately).** Running the full winner-vs-each grid on the k-sweep (15 tests: 5 pairs × 3 axes) "found" two significant results that are almost certainly false positives: k=5 beating its neighbors k=6 (faith +0.35, p=0.026) and k=7 (faith +0.40, p=0.045) while being indistinguishable from the distant k=10 (p=0.37). That non-monotonic shape is what noise looks like, not what a retrieval-width effect looks like - and at 15 tests with α=0.05 one expects ~0.75 false alarms per grid. Bonferroni (0.05/15 ≈ 0.003) kills both, and grazes even the one honest-looking signal: **k=5 vs k=1 completeness +0.68, p=0.009** - a single chunk visibly hurts answer completeness, the only defensible per-pair conclusion in the sweep. Revised k-sweep verdict: avoid k=1; between k=3 and k=10 no difference is provable at n=100; keep k=5 as the cheap middle. Lesson stacked on the lesson: intervals fixed the point-estimate problem, and immediately exposed the next one - the more comparisons you look at, the stricter your threshold must be, or something "significant" will always turn up.

**Full-matrix verdict (all 15 k-pairs × 3 axes, 45 tests).** One coherent signal survives scrutiny: **k=1 hurts completeness**, losing to every k≥5 (+0.51..+0.68, p=0.009..0.047) with k=3 pointing the same way - four same-direction significant pairs plus a plausible mechanism outweigh the fact that no single pair clears Bonferroni. The two k=5-beats-neighbors faithfulness hits are non-monotonic (k5≈k10), mechanism-free, and dismissed as multiple-comparison noise. Everything between k=3 and k=10 is a statistical plateau on every axis at n=100. Final answer to "which k is best": provably not 1; within 3..10 undecidable; k=5 stays as the plateau middle with the best point estimates (all deltas vs k=3 positive but unproven). To actually separate 5 from 3, the lever is sample size, not a cleverer test: n≈500-1000 shrinks the CIs ~3x - exactly the multi-fidelity step (search at 100, confirm at 1000) the experiment design already anticipates.
