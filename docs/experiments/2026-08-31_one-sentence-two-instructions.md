# 2026-08-31 - One sentence, two instructions, and the arm that told them apart

## Setup

**Question** Does the judge penalise a Russian answer for being Russian, rather than for what it
says? `judge.completeness.v2` carries a clause about judging across languages; `judge.faithfulness.v2`
and `judge.relevance.v2` do not. Russian faithfulness is about a point lower than English on the
same question (-0.939 for llama, -0.772 for gemma, paired on the shared parent, n=820), while
completeness holds (-0.184 and -0.070). The clause is the obvious suspect.

**Kind** rejudge · **source** `grid_llama31_8b_ru_plain` (823 rows, single_shot, no reranking,
llama3.1:8b) · **judge** qwen2.5:7b at temperature 0 · **corpus** `clean_1024`

- nothing is generated: the arms re-score answers that already exist, and `same_answers` is true
  for all three pairs, 823 of 823 byte-identical
- v3 differs from v2 by one inserted sentence in each of two prompts, quoted under Result
- a **repeat arm** re-scores the same rows with v2 in today's residency. Without it the comparison
  would be against verdicts made on 29.08, and the judge drifts between passes
- the predictions and both degeneracy conditions were written to the arc log before the run

## Result

```
pair                          faithfulness              relevance                completeness
v3 vs repeat of v2        +0.2539 [+0.1725,+0.3366]  +0.3293 [+0.2454,+0.4156]  +0.0036
  (both arms today)          123 better, 48 worse       153 better, 48 worse      3 rows moved

repeat of v2 vs               -0.0972 [-0.1677,-0.0316]  +0.0243 [-0.0255,+0.0765]
  the 29.08 source             the judge's drift over two days
```

The effect is clear of zero on both rejudged axes. Completeness, which was not rejudged, does not
move: 3 rows of 823, which is the control on the arms themselves.

## The degeneracy condition fired

Written before the run: "degenerate if the gain exceeds 1.5 of a point **or if the share of tens
jumps**, because that is not `stopped penalising the language`, that is `got kinder to everything`".

```
share of tens        faithfulness    relevance
repeat of v2             0.1373        0.6355
v3                       0.2053        0.7412
share >= 8               0.2078        0.7351   ->   0.2722   0.8117
```

Tens on faithfulness went up by half. The second condition is what caught it, and it was written
because the previous entry's falsification condition had been one-sided and caught nothing.

## The prediction that failed is the one that gave the answer

Prediction 3 said relevance would move **less** than faithfulness, since its gap is smaller
(-0.405 against -0.939). It moved **more**.

That is decisive, because only one of the two axes has a language mismatch at all. Faithfulness
compares a Russian `RESPONSE` against an English `CONTEXT`: the corpus is almost entirely English,
so every row of this run is a cross-language case. Relevance compares a Russian `ANSWER` against a
Russian `QUESTION`: there is no mismatch anywhere in the set. A mechanism that stops penalising
the wrong language cannot move relevance, and relevance is what moved most.

## The cause is the intervention, not the judge

The inserted sentence carries two instructions rather than one:

> **Judge by meaning, not wording**, and across languages (the RESPONSE and the CONTEXT may be in
> different languages); answering in another language is not itself ungrounded.

The first half is a general instruction to be lenient. The second is the hypothesis. The phrase was
copied whole from `judge.completeness.v2` to keep the voice of the existing prompts, and that broke
the rule this round was written under: one thing changes at a time. On the evidence, what was
measured is the first half.

## Decision

**Not adopted, and the hypothesis is not refuted either: it was not tested.** v3 was deleted with
the round it served, prompts and rows both, so the version number does not sit in the registry
looking like an option. The 823 judged rows still carry `judge_faithfulness: 3` in their snapshot,
and the sentence they were judged by is quoted above and lives in commit `19565fc`. **Number 3 is
not to be reused for these two purposes**, or those rows would silently start pointing at another
prompt; the next round starts at v4.
The next round splits the sentence: one arm carrying only "across languages", one carrying only
"by meaning, not wording", both as rejudges of the same source, so neither costs generation.

**Correction, same day.** The paragraph above says number 3 is not to be reused, and that held only
until the round it served was over. The owner asked for a registry without gaps, so the rows that
referenced 3 and 4 were deleted and the numbers freed: version 3 now carries the surviving half,
"Judge by meaning, not wording." The verdicts of both retired versions, and their prompt text, live
in `datasets/measurements/retired_judge_prompts_v3_v4.json`, and every number on this page
reproduces from that file. Read this entry through the file, not through the registry.

Sizing is decided from the effect rather than by habit: the observed differences are 0.25 to 0.33,
and the paired half-width at n=400 is about 0.126, so the next round reads on 400 rows instead of
823. The full set buys precision this question does not use.

## Caveats

- **The judge's drift is not a constant.** Three independent measurements now: +0.2442 between two
  identical single-shot passes (29.08, n=823), +0.52 between two identical agent passes (31.08,
  n=50), and -0.0972 between the 29.08 source and today's repeat of the same prompts. The
  half-width of 0.088 describes one pair of passes, not a property of the judge, and quoting it as
  a constant is a mistake this entry makes visible rather than repeats.
- Language slips are not what this measures. On this source only 3 answers of 823 slipped into
  English by the prose reading; the cross-language case here is the corpus, not the generator.
- One source, one language, one generator. Whether the same insert moves an English run is a
  separate arm and was pre-registered as payable only if the Russian side moved, which it did.
