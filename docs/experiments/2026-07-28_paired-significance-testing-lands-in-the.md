# 2026-07-28 - Paired significance testing lands in the aggregator (the audit answered)

An external audit named the sharpest weakness of the lab: point estimates without uncertainty. The
RRF winner k=5 beat k=10 by 0.0487 against 0.0484, and rerank deltas of +0.2 to +0.4 on a 0-10
scale could live entirely inside noise. Every comparison here is paired, the same question answered
in both configs, so paired statistics are nearly free.

## Setup

**Set** `paraphrased_ru` (n=100 per arm) · **corpus** pre-variant era, post-ablation · **judge** `qwen2.5:7b`, numeric 0-10

The aggregator now computes `composite.pairwise` for every experiment: winner against each other
value, per generation axis, with the mean paired delta, a bootstrap 95% CI (10k resamples, fixed
seed) and a Wilcoxon signed-rank p-value. Rank-based, which is right for ordinal judge scores with
outliers; zero-delta series short-circuit to p=1. Exposed as `pairwise_stats(run_a, run_b)` and
stored in experiment `results`.

The tolerant numeric parser is gone with it: after the full re-judge of the categorical era (1442
logs re-scored), a non-numeric verdict in the DB is a bug and fails loudly instead of being skipped.

## Result

| comparison | axis | delta | CI95 | p | verdict |
|---|---|---|---|---|---|
| 70b vs 8b (single_shot k5) | faithfulness | +0.38 | [-0.04, 0.81] | 0.034 | borderline |
| 70b vs 8b | relevance | -0.04 | [-0.68, 0.60] | 0.58 | noise |
| 70b vs 8b | completeness | -0.10 | [-0.50, 0.31] | 0.49 | noise |
| k=5 vs k=10 (agent baseline) | faithfulness | +0.19 | [-0.19, 0.57] | 0.37 | indistinguishable |
| k=5 vs k=10 | relevance | -0.04 | [-0.49, 0.40] | 0.98 | indistinguishable |
| k=5 vs k=10 | completeness | +0.07 | [-0.33, 0.48] | 0.89 | indistinguishable |

## What this does to our own conclusions

- **k=5 against k=10 is a coin flip on every axis.** The k=5 default stands, but the justification
  is cost rather than proven quality: at indistinguishable quality, feed the generator half the
  chunks. The earlier "k=5 wins the composite" phrasing overstated the data.
- **The 70B's faithfulness gain is marginal.** Wilcoxon says significant at p=0.034, the bootstrap
  CI grazes zero. "23x latency for a borderline single-axis gain" is a stronger version of the
  stay-on-8b decision.
- Rerank per-k deltas (completeness +0.2 to +0.4) remain untested pairwise; suggestive until then.

## The multiple-comparisons trap, found immediately

Running the full winner-against-each grid on the k-sweep (15 tests: 5 pairs by 3 axes) produced two
significant results that are almost certainly false positives: k=5 beating its neighbours k=6
(faithfulness +0.35, p=0.026) and k=7 (+0.40, p=0.045) while being indistinguishable from the
distant k=10 (p=0.37). A non-monotonic shape like that is what noise looks like, and at 15 tests
with alpha 0.05 one expects about 0.75 false alarms per grid. Bonferroni (0.05/15, about 0.003)
kills both and grazes the one honest-looking signal, k=5 against k=1 on completeness (+0.68,
p=0.009).

Over the full matrix (all 15 k-pairs by 3 axes, 45 tests) one coherent signal survives: k=1 hurts
completeness, losing to every k of 5 and above (+0.51 to +0.68, p=0.009 to 0.047) with k=3 pointing
the same way. Four same-direction significant pairs plus a plausible mechanism outweigh the fact
that no single pair clears Bonferroni. The two k=5-beats-neighbours hits are non-monotonic,
mechanism-free, and dismissed.

## Decision

Every experiment now reports intervals and Bonferroni-corrected flags alongside the composite.
On k: provably not 1, undecidable between 3 and 10, k=5 stays as the plateau middle with the best
point estimates. Separating 5 from 3 is a sample-size question rather than a test question: n of
500 to 1000 shrinks the CIs about threefold, which is the multi-fidelity step the experiment design
already anticipates (search at 100, confirm at 1000).

## Caveats

- the family of tests here was drawn after the runs, so the correction describes the grid rather
  than justifying any single verdict in it
- bootstrap and Wilcoxon both assume the pairing is real; a run whose questions drifted between arms
  would break that silently, which is why the run snapshot pins the set
