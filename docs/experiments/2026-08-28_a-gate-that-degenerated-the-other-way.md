# 2026-08-28 - A gate that degenerated the other way

Reranking every question costs 168 ms and, on the English set, does not pay for itself. The obvious
next move is a gate: rerank only where the fusion looks unsure. This entry measures that gate and
closes the idea, and the interesting part is how the falsification condition failed to fire.

## Setup

**Sets** `paraphrased_v2_ru` (n=823) and `paraphrased_v2` (n=820) · **corpus** `clean_1024` ·
**judge** none, retrieval only

- simulated over recorded candidate lists rather than run: the gate decides from the fused list, so
  every arm is a re-read of the same retrieval
- two candidate signals: `d1`, the distance to the nearest chunk, and `gap`, how far the second
  candidate sits behind the first
- the threshold is chosen **by rule on half A** (maximum paired delta) and reported on half B. No
  number from B takes part in choosing it
- the half-width was computed before the run: on the English set, sd of paired rerank deltas 0.2959
  over n=820 gives 0.0203 on the whole set and 0.0286 on a half, so nothing under about 0.03 is
  measurable here and predicting a point inside that band is meaningless

## What was predicted, before the run

1. `d1` beats `gap` on AUC. On the Russian set the auditor had 0.718 against 0.669, and the argument
   is that `gap` measures how alike two neighbours are while `d1` measures whether anything close
   was found at all
2. the gated rerank comes out of the minus on English, where unconditional reranking reads
   -0.0083 [-0.0283, +0.0125]: point estimate on B not below zero, interval through zero
3. the A-to-B gap is of the order of the half-width, that being the price of choosing a threshold on
   the same half it is measured on. On Russian it had been 0.046
4. the share of questions sent to the reranker lands between 0.3 and 0.8. **If it goes over 0.9 the
   gate has degenerated into unconditional reranking and there is nothing to compare**

## Result

```
paraphrased_v2 (English, n=820), clean_1024, threshold chosen on A, reported on B

unconditional rerank    A  -0.0068 [-0.0344, +0.0227]   B  -0.0097 [-0.0389, +0.0189]

d1  (distance of the first)   AUC 0.728   threshold on A 0.4312   2% of questions reranked
                        A  +0.0035 [+0.0000, +0.0087]   B  +0.0033 [+0.0000, +0.0086]

gap (lead of the second)      AUC 0.310   threshold on A 0.0075   32% of questions reranked
                        A  +0.0105 [-0.0075, +0.0278]   B  +0.0074 [-0.0115, +0.0267]
```

Two predictions hit, two missed. `d1` beats `gap`, and the point on B is not below zero. The A-to-B
gap did not appear, and the share went to 2% rather than 0.3 to 0.8.

The third miss follows from the fourth: an A-to-B gap is the price of fitting a threshold, and there
was nothing to fit. The optimum on half A did not choose a threshold, it chose a refusal.

## The falsification condition was written one-sided, and that is the finding

The pre-registration said the gate is degenerate if the share goes **over 0.9**, meaning it reranks
everything. It went the other way, to 0.02, and the condition as written did not catch it.

A gate has two degenerate states, always and never, and the second one is the dangerous one because
it looks like a win: +0.0033 with an interval touching zero reads better than the -0.0097 of
unconditional reranking, and the reason is only that the gate has switched reranking off. The
default did that already, for free, with no threshold and no branch.

**A degeneracy condition is written from both sides.** That rule came out of this run and is kept
with the arc's other reading rules.

## An aside worth carrying: the signal is inverted between languages

`gap` reads AUC 0.669 on Russian and 0.310 on English. Not weaker, inverted: on English a larger
lead over the second candidate predicts a *wrong* first answer. One quantity, two languages, two
directions. That is an argument against carrying thresholds between languages, and it applies to
every threshold in the config that was calibrated on one of the two sets.

## Decision

**The idea is closed, not deferred.** On Russian the gate loses to unconditional reranking on the
same half it is reported on: +0.0179 [-0.0089, +0.0451] against +0.0294 [+0.0001, +0.0595] on half B.
(Unconditional reranking over the whole Russian set is +0.0454 [+0.0250, +0.0670]; that is the number
the config quotes, and it is not the one to compare a half against.) On English it
beats unconditional reranking by switching it off almost always, which the default already does. A
config key, a per-language threshold and a second branch in the retrieval path are not worth
+0.0033 on half a set.

## Caveats

- This says nothing about reranking being harmful on English. Unconditional reranking there is
  -0.0097 [-0.0389, +0.0189]: the set shows no harm as clearly as it shows no benefit. What settled
  reranking as a default was measured on answers rather than ranks, in the generator grid entry of
  2026-08-29.
- Both signals are read off the fused list, so the entry says nothing about a gate built on
  something else: a classifier on the question, or the cross-encoder score itself.
- The simulation reuses recorded candidates, which is what makes it cheap and also what limits it:
  it cannot see an effect that would change which candidates were retrieved in the first place.
