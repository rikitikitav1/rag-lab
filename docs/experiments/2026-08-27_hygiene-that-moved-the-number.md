# 2026-08-27 - Corpus hygiene that moved the number, and four instruments that were lying

Cleaning the sources moves retrieval on its own, before any change of splitter: parse the
frontmatter, take the heading path from a declared root, drop the junk by rule, and measure
`baseline → clean_1024` with nothing else changed.

## Setup

**Set** `paraphrased_v2_ru` (n=823) · **corpus** `clean_1024` against `baseline` · **judge** none, retrieval only

- criterion set grown in place from 344 questions so the old rows keep their ranks
- halves are a pure function of the question id; the winner is chosen on A and reported on B
- exact search on both arms (`enable_indexscan = off`), because rebuilding the hnsw index moves the
  ranking and the primary criterion must not depend on that
- paired bootstrap intervals over per-question deltas of the reciprocal rank

## Result

```
baseline -> clean_1024, 823 questions, exact search

section  all +0.0435  CI [0.0298, 0.0575]   better 203, worse 84
         A   +0.0373  CI [0.0156, 0.0590]   n=401, 168 repositories
         B   +0.0494  CI [0.0312, 0.0676]   n=422, 166 repositories

file     all +0.0286  CI [0.0159, 0.0410]   better 135, worse 63
         B   +0.0264  CI [0.0099, 0.0435]
```

None of the six intervals touches zero. Reported on half B by pre-registration: **+0.0494 by
section, +0.0264 by file**.

## What the day actually cost: four instruments that were lying

The number above was taken three times, because three separate instruments turned out to be
measuring something other than their name.

**`ef_search = 200` was not a depth of the index, it was the absence of one.** pgvector prices the
hnsw path linearly in `ef_search` while sorting the whole table costs a fixed amount, so above
about ef 197 on this corpus the planner drops the index and sorts exactly instead. The recall
ladder then compared exact search with exact search, which is why 200 and 400 both read exactly
1.0 and why `required_ef` came out at 200. Production was answering every question with a
sequential scan over 13068 vectors: right answers, 65 ms instead of 1 ms, nothing in the logs. The
crossover is a property of the row count and moves up as the corpus grows, so 100 keeps the same
plan through growth where 200 would silently switch back. Every rung of the ladder now asks for its
plan first and refuses one that holds no index scan.

**recall@20 was the wrong question.** It counts neighbours the graph failed to find, and a
neighbour lost at rank 18 costs the answer nothing. Measured properly, paired, on the same 823
questions: at ef 100 the index loses the gold section in **none** of them on `baseline`, and the
delta of section MRR against exact search is +0.0005. So the gate moved off recall and onto what
the depth costs the metric, with a threshold set to what the set can resolve (half the interval is
0.0036, so 0.005 would have been a coin toss). One question is refused by the new hard gate rather
than by the mean: a Russian paraphrase with no distinctive term in it, whose whole neighbourhood
sits in a band of 0.028 with `Vue.js` and a Redis command above the right answer. The vector leg
has no signal there at all; the hybrid found it anyway.

**The cut was not reproducible from its own code.** The regex called any line starting with two
hashes a section, so a comment inside a fenced block cut an example in half; 17 such lines exist in
the corpus. Worse, the intro of a file lost its first line whenever that line began with a hash,
which took the first entry out of **215 of 326** cheatsheets files. Both were fixed, both changed
the rows, and the reproduction check was looking at `baseline`, the one variant frozen by
definition. It asks every indexed variant now and compares the text of each chunk, because
fourteen of the sixteen sources that changed kept their row counts exactly.

**A gate that cannot fail is not a gate.** Two of the three hard gates on the coverage report were
arithmetic identities: `prefix_broken` compared the rendered prefix against the string the prefix
was rendered from, and `section_coverage` asked whether a chunk has a section when the cutter
writes one on every chunk. The first is gone rather than repaired, the second now asks whether the
source has structure under its root, and a metric with nothing to measure abstains instead of
returning zero, because a zero passes a gate and earns its full weight in the score.

## The re-index, and a prediction written before it

Restoring the lost headings and moving to the standard parser meant the indexed rows no longer
matched the code, so the point had to be taken again. Five predictions were written down first. The
sharp one: of the 823 criterion questions, **zero** come from the three sources whose cut changed,
so the gold chunks are the same rows and only a competitor can move.

What came back: the head delta identical to the fourth decimal, the old cut against the new one
showing **no movement at all by file** and two questions by section, each one position, 12109 rows
against a predicted 12098 ± 20, and the reproduction check green with zero differing sources of
177. The one prediction that missed was the sign of a point estimate inside an interval that covers
zero, which is what noise looks like.

## Decision

`ef_search: 100`. The ladder and its gates live in the config beside the value they judge. The gate
on the index is the loss it causes the answer, plus a hard rule that a question the exact arm
answered may not be absent from the index arm. `clean_1024` is re-indexed and reproducible, and it
is point zero of the ceiling grid rather than a variant standing beside it.

## Caveats

- the thresholds on the coverage report and on the index cost are set by hand from what the corpus
  reads today; they are calibration, not law
- `clean_1024` chunks are longer than `baseline` ones by the length of the prefix, because the
  ceiling is declared on the body. More text per chunk raises the prior chance of holding the gold
  span independently of hygiene, and that confound is in the number above
- the whole criterion set comes from the interview repositories, so nothing here speaks for the
  three documentation sources; the veto run on agent pools is what will have to speak for them
