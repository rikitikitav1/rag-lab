# 2026-08-30 - The ceiling that changed nothing, and the rule that decided anyway

Doubling the chunk ceiling from 1024 to 2048 was the change the corpus shape argued for: at 1024
the size counter cuts 69% of the sections, so the structural cut barely gets to act. This entry
asks whether that argument survives a measurement, and what a comparison decides when it does not
resolve.

## Setup

**Sets** `paraphrased_v2_ru` (n=823) and `paraphrased_v2` (n=820) · **veto set** `veto_v1` (n=270)
· **corpus** `cap_2048` against `clean_1024` · **judge** none, retrieval only

- both variants hold the same 1002 files; `cap_2048` holds 7,102 chunks against 12,102
- `cap_2048` was dropped from the table and from the config once it lost. Its policy, so this
  entry can be rebuilt from itself: `{ chunker: rooted, max_chunk_size: 2048, ceiling_on: body }`
- exact search on every arm (`enable_indexscan = off`): rebuilding the hnsw index moves the
  ranking, and the primary criterion must not depend on that
- paired bootstrap intervals over per-question deltas of the reciprocal rank, eight seeds
- halves are a pure function of the question id; the winner is chosen on A and reported on B
- **the decision rule was written before the second point was indexed**, in the arc log, and is
  quoted under Decision. That timing is the whole point of this entry

## Result

```
clean_1024 -> cap_2048, exact search, pool 100

criterion, section   all  +0.0070  CI [-0.0064, +0.0208]   147 better, 117 worse   n=823
                     A    +0.0142  CI [-0.0071, +0.0355]   (choice)
                     B    +0.0002  CI [-0.0181, +0.0178]   (report)
criterion, file      all  +0.0035  CI [-0.0072, +0.0144]   n=823

english, section     all  +0.0011  CI [-0.0113, +0.0145]   139 better, 115 worse   n=820

veto, section        all  -0.0180  CI [-0.0365, -0.0006]    32 better,  46 worse   n=270
veto, file           all  -0.0131  CI [-0.0303, +0.0025]    18 better,  32 worse   n=270
```

Every interval on the criterion crosses zero, on both halves and both languages, and every one of
them is stable across eight resampling seeds. **The doubled ceiling is indistinguishable from the
1024 one on the questions the criterion is made of.**

## The veto leans down, and by our own rule it does not fire

The veto set covers the four families the criterion cannot see (`cheatsheets`, `redis-doc/docs`,
`notes`, `system-design-primer`), which are 37% of the corpus and had no questions at all before
this arc; how it is built and what it may decide is its own entry,
[The questions the criterion cannot see](2026-08-30_the-questions-the-criterion-cannot-see.md).
Its rule is asymmetric on purpose: the winner is not required to gain there, only to not
lose, and the test is that the interval does not lie entirely below zero.

By section it does lie below zero at seed 0, `[-0.0365, -0.0006]`. Across eight seeds the upper
bound is negative on seven and reads `+0.0001` on the eighth:

```
upper bound by seed:  -0.0006  -0.0002  -0.0008  -0.0008  -0.0003  -0.0003  +0.0001  -0.0003
```

A bound that changes sign when the resampling changes is a parity, not a result: that rule was
established on the rejudge bench earlier the same week, and applying it here costs us the more
dramatic reading. The veto does not fire. The numbers are printed as they are so that
`[-0.0365, -0.0006]` is never later quoted as a refusal.

Descriptively, all eight family-level cells point down and only one excludes zero:

```
section   cheatsheets  n=80  -0.0280 [-0.0683, +0.0090]   12 better, 12 worse
          notes        n=80  -0.0220 [-0.0489, -0.0013]    4 better,  7 worse
          redis-doc    n=80  -0.0038 [-0.0414, +0.0347]   11 better, 19 worse
          primer       n=30  -0.0186 [-0.0645, +0.0256]    5 better,  8 worse
file      redis-doc    n=80  -0.0285 [-0.0688, +0.0065]    6 better, 14 worse
```

One consistent direction, not enough force. The veto set resolves differences of about 0.03 MRR
at n=270, and this is under that.

## Read again through the reranker we actually serve

The criterion runs on the bare hybrid search so that it measures the cut and not the pipeline, but
what serves answers reranks. A ceiling that helps the bare search and hurts after reranking would
have passed unnoticed, so both arms were read again with `--rerank-top 20`. This reading is
descriptive and was not part of the decision.

```
clean_1024 -> cap_2048, with reranking, section, n=823
   all  -0.0047  CI [-0.0198, +0.0094]     A  -0.0098     B  +0.0001
```

Still through zero, and the point estimate has flipped sign against the bare reading (+0.0070 to
-0.0047). Two readings of the same comparison disagreeing about the sign of a point estimate is
what "no difference" looks like from the inside.

The same runs answer a question nobody asked, and it is the largest number in this entry:

```
reranking within one variant, paired, section, n=823
   clean_1024  +0.0454  CI [+0.0250, +0.0670]   190 better, 135 worse
   cap_2048    +0.0336  CI [+0.0121, +0.0559]   201 better, 145 worse
```

Reranking is worth about +0.045 of section MRR. The entire choice between chunk ceilings was
+0.0070 with an interval through zero. One switch that was already on outweighs the corpus variant
this entry spent a day indexing and measuring by roughly sixfold. The two variants' reranking
intervals overlap, so "reranking helps the smaller ceiling more" is an observation, not a result.

## 4096 was refused before it was measured

The grid was planned as 2048 and 4096. The larger point never ran, and the reason is not retrieval.

Context is five chunks joined. Generation asks for 1024 tokens of the 8192-token window, leaving
7168 for the prompt. Measured on 1,643 rows of the generator grid, the worst observed ratio is 2.20
characters per token (dense code tokenizes worse than prose) against a mean of 3.42:

```
ceiling   worst-case context    at 2.20 chars/token   plus 1024 for the answer   window 8192
1024        6,230 characters         2,832 tokens          3,856                 4,336 spare
2048       11,350                    5,159                 6,183                 2,009 spare
4096       21,590                    9,814                10,838                 OVERFLOW
```

At the mean ratio 4096 fits, at 7,337 of 8,192. So it would truncate on dense contexts and not on
others, and ollama truncates without saying so: there would be numbers, and they would be wrong on
a subset nobody could identify afterwards. The owner refused the point rather than raising the
window (the KV cache roughly doubles, and the card is at 6,382 of 8,188 MiB with the reranker on
it) or lowering `k` for that arm alone (`k` is part of the pipeline; an arm at another `k` differs
in two things).

`unbounded` died to the same gate and more decisively: the longest section body in the corpus is
36,794 characters, so a single chunk of it is about 16,700 tokens, twice the whole window.

## Decision

**`clean_1024` stays the corpus point, and what decided it is the tie-break rule, not the
numbers.** The rule was written into the arc log before `cap_2048` was indexed, and its relevant
clause reads: when the candidates' intervals overlap, the smaller ceiling wins. The argument given
there, in advance, was that the smaller ceiling is cheaper in context and further from silent
truncation, which the 4096 arithmetic above then made concrete.

This is the point of pre-registering a tie-break. Written afterwards, "the smaller ceiling wins on
a tie" would be indistinguishable from rationalising the result that happened to arrive.

Also decided:

- `cap_2048` was dropped from the table the same day, by the owner's decision, and its declaration
  went with it: the policy is quoted in Setup above, which is what makes this entry rebuildable
  with one job. What the deletion cost is in the correction below
- `cap_4096` and `unbounded` are refused by the context guard, which is now stated as a rule:
  `k × (ceiling + longest prefix) / 2.20 + generation budget <= OLLAMA_CONTEXT_LENGTH`, checked at
  the door from the config, before anything is indexed
- the prior that motivated the grid is recorded as **not confirmed**: the section-length
  distribution said the ceiling was where the gain was, and doubling it moved nothing

## Caveats

- **The grid answers "not worse", not "which is better".** The interval half-width on 823 questions
  is about 0.014 MRR by section and neighbouring ceiling points differ by that order, so this was
  written down before the run rather than discovered after it.
- **The veto's absolute numbers are a floor, not a measurement.** Each question carries one marked
  file, and some of its questions could be answered honestly from a different family. The bias is
  constant between arms and cancels in the paired delta; the absolute `hit@k` does not get to be
  quoted as retrieval quality. Full description: `notes/reference/veto_set.md` (internal).
- **Indexing a third variant moved the search depth**, and this is the entry's most portable
  finding. The depth at which the planner abandons the hnsw walk turned out to be a property of the
  **variant**, not of the table: `cap_2048` with 7,102 rows walks its smaller partial index past
  1000, while `clean_1024` (12,102) and `baseline` (13,068) stop at 248 and 259 in the same table.
  Deleting a variant moves those crossovers **down**, because a smaller table is cheaper to read
  sequentially: at two variants and 25,177 rows the crossover was measured at ~197, below the depth
  now pinned. Cleaning up losing variants is therefore not a tidying operation, it is a change to
  the served search depth.
- Nothing here says anything about generation. Both variants were read by rank alone, no generator
  and no judge were involved, and a ceiling that changes what an answer looks like without changing
  where the chunk ranks would be invisible to this entry entirely.


## Correction, measured on 2026-08-30, same day

The Caveats section predicted that deleting a variant would move the depth crossover down, and
estimated it would land near 197 from a two-variant reading recorded on 28.08. The owner chose to
delete `cap_2048` and re-measure immediately rather than at the end of the arc. The prediction was
right in direction and badly wrong in size.

```
before   3 variants   32,272 rows   8,959 pages   crossover  baseline 259, clean_1024 248
after    2 variants   25,170 rows   4,700 pages   crossover  baseline 126, clean_1024 123
```

The estimate missed because a `DELETE` of 7,102 rows was not what shrank the table. `VACUUM FULL`
was, and it reclaimed bloat that had nothing to do with this variant: pages fell by 47% while rows
fell by 22%, and the table went from 613 MB to 380 MB.

At 123 the depth pinned that morning, 200, no longer walks the index. The preflight refused, which
is the behaviour it was pinned for.

**The rewrite made the shallower rung better than the deeper one had been**, because `VACUUM FULL`
rebuilds the hnsw graphs:

```
ef 100, clean_1024, paraphrased_v2_ru, n=823
  before the rewrite   section MRR against exact +0.0011 [0.0002, 0.0026]   1 question lost   recall@20 0.9713
  after  the rewrite   section MRR against exact +0.0000 [0.0000, 0.0000]   nothing lost      recall@20 1.0
```

Depth is now pinned to 100. Both readings are honest and they describe different graphs, which is
the part worth carrying forward: **an hnsw number is a fact about a graph, not about a corpus**, and
any operation that rebuilds the index invalidates every recorded hnsw measurement while leaving
every exact-search measurement untouched. The grid results above are exact-search and stand
unchanged.

Two things follow for anyone reading this later. A depth pinned in the config is only as stable as
the physical table under it, and the audit rather than the number is what makes it safe. And a
`VACUUM FULL` is not maintenance here: it is a change to the retrieval path, and it belongs in an
entry rather than in a shell history.
