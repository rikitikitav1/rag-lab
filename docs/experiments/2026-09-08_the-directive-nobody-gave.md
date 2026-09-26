# 2026-09-08 - Answer-language directive fix, and the judge's penalty for Russian

The stand answers Russian questions about an English corpus. Nothing in it ever told the model which
language to answer in. The single-shot path got away with it and the agent did not, and fixing the
one line that caused it turned out to be a way of pricing a defect in our own judge.

## Setup

**One line, two paths.** Both answering paths appended a language directive to the prompt only when
a run had explicitly asked for a language, which no run ever did. `chat.answer_from_rows` resolved
the language a line earlier and passed it to the record instead of to the model. The fix builds the
directive from the resolved language, always · **runs** `arc3_agent_baseline` (before),
`arc4_lang_after` (after, the same 300 questions), `arc4_baseline_rejudged` (the before arm judged
again, in the after arm's residency) · **generator** `llama3.1:8b`, agent path, `max_hops 4`, `k 5`,
no rerank, variant `clean_1024` · **judge** `qwen2.5:7b`, our prompts, residency 2516 for both arms
· **detector** `db.detect_language`, function words before the alphabet, the same rule that picks
the search config

## Why single-shot got away with it

Measured over every answered row in the declared population, the share of answers coming back in the
language of the question:

```
single shot, grid runs      100%     823 of 823
agent, arc3_agent_baseline   73%     112 of 154
agent, arc3_ported_after_split 74%   114 of 154
```

Single-shot puts the retrieved context and the question in one message, so the Russian question is
the last thing the model reads before answering. The agent puts the question at the head of a
transcript that then fills with English tool output, and by the time it answers, the corpus has been
talking to it for four hops. Neither path instructed anything; one of them was standing close enough
to the question for that not to matter.

A first count put the agent at 60% and the corpus-wide drift at 13.2%. That count was a hand-rolled
ratio of Cyrillic to Latin characters, and it read a Russian answer full of code and English terms as
English. The stand already had a detector, and it is the same one the pipeline uses to choose the
answer language. The corrected numbers are the ones above.

## What was predicted before the run

Written into the arc log before the run started, because a prediction made after a number is a
description:

1. `language_match` rises above 0.95. Below 0.85 would mean the directive does not survive an English
   transcript, and its place is the last message rather than the system prompt.
2. **Our three axes do not rise.** The judge was already known to lose about a point on Russian at
   the same meaning, and the arm was about to answer in Russian far more often.
3. The share of narrated tool calls does not move. The directive has nothing to do with calling
   tools, and if it moved that too, the prompt is sensitive to any added line rather than to this one.

## What the run gave

```
language_match      0.727 -> 0.987        paired +0.265 [0.197, 0.34] over 147 questions
narrated calls      51 of 300 -> 56 of 300
```

Predictions 1 and 3 held. The directive holds in the system prompt across four hops of English tool
output, so it does not need to be repeated at the end.

## The floor, measured rather than assumed

The before arm had been judged two days earlier, the after arm today. This stand has measured what
that costs: across a model reload the same judge moves 14% of its scores on byte-identical input. A
delta of a quarter of a point between two arms judged in different residencies is not readable.

So the before arm was copied and judged again, in the after arm's residency, and the two judgings of
the same answers give this contrast its own floor:

```
faithfulness   +0.000 [0.000, 0.000]     0 rows of 99 moved
relevance      -0.010 [-0.131, 0.121]    9 moved
completeness   -0.061 [-0.152, 0.000]    2 moved
```

Zero on the axis that matters here. Worth saying plainly: that is a property of this pair, not of
one residency. The day before, one pair repeated 144 rows with no change at all and another moved 4
rows of 50 inside the same load.

## Prediction 2, and where the drop lives

Both arms judged in residency 2516, 99 pairs on the corpus pool:

```
faithfulness   -0.253 [-0.687, 0.192]
relevance      -0.232 [-0.646, 0.192]
completeness   -0.323 [-0.717, 0.071]
```

No axis rose. None of the bands excludes zero either, so on its own this says only that the fix did
not buy anything our own metrics can see.

Splitting is where it becomes a number, and which split is quoted matters more than the number.
The cut to quote is the one visible in the record **before** the change: rows where the baseline
answered in a language other than the one asked. That is a covariate of the arm as it stood, not a
property of the result.

```
answered off language before, 37 rows   faithfulness  -0.838 [-1.459, -0.243]  p 0.015
                                        completeness  -0.676 [-1.351,  0.054]
                                        relevance     -0.081 [-0.811,  0.703]

changed language, 35 rows               faithfulness  -0.971 [-1.600, -0.371]
kept its language, 64 rows              faithfulness  +0.141 [-0.438,  0.719]
```

The second cut is the one first published, and it was selected by the outcome of the change: you
can only see which rows switched after the fix worked. Thirty-five of the 37 declared rows are in
it, so the two nearly coincide, but only the first can be named without seeing the result.

The whole of the faithfulness drop sits on the rows that were answering in the wrong language.
Where the language was already right, the axis did not move.

## Minus 0.838 against 0.964, and why that is agreement rather than replication

The day before, a probe measured the same thing by a different construction: it took one answer and
asked the model to restate it in two languages without changing what it said, then judged both. That
gave **0.964 [0.645, 1.282]**.

The first version of this entry called the two a replication. That was wrong, and the correction is
worth more than the number. This run restated nothing. The two arms hold different answers, and on
half the pairs different contexts as well: the directive sits in the system prompt, so it changes
the transcript from the first step and therefore the tool calls too. "Exactly one thing differs" is
true of the code and false of the pairs. The measured drop is the judge's penalty **plus** the real
difference between what the model writes in Russian **plus** the difference in what retrieval
returned, in unknown proportions. Two point estimates landing a tenth apart inside a band a full
point wide is agreement, not a second instrument confirming the first.

## The night after: the judge's penalty does not reproduce, and the reading is now open

The cheap separator the auditor proposed was run on the 37 declared rows: take each row's own
answer, restate it in both languages without changing what it says, judge both. If the drop is the
judge charging for Russian, this returns about 0.84.

```
arc4_lang_after, the 37 declared rows   +0.216 [-0.135, 0.595]   control 0.919, in regime
arc4_baseline_rejudged, 38 english      +0.184 [-0.395, 0.711]   control 0.974, in regime
arc4_baseline_rejudged, 38 russian      +0.184 [-0.263, 0.632]   control 0.895, OUT OF REGIME
arc3_interview_independent (07.09)      +0.964 [0.645, 1.282]    control 0.991, in regime
```

The third line is not read: the control missed its floor by five thousandths, and the rule that
threw out a number on 07.09 is the same rule here. Its agreement with the line above it is a
remark, not evidence.

Two readable probes out of three now say the penalty is around 0.2 with a band across zero, and the
0.964 stands alone on one population. So the sentence below, written earlier the same day, is no
longer supported: what the judge charges for Russian looks like a property of the population it was
measured on. And the -0.838 loses its main explanation. It is at least as well explained by the
generator writing less grounded answers in Russian, and by the retrieval that changed with the
transcript.

## What it means for everything measured before

The defect was read as the judge's, and that reading is now open (see the section above). It sits on
the axis this stand uses to mean "grounded". Every number taken here on a population mixed by language carries it in an unknown
proportion, including the agent against single-shot comparison, because neither arm was instructed
and both drifted at rates we had not measured.

Fixing the judge's prompt would end the comparability of everything judged so far, which is why it is
a decision and not a task. The cheap half is done: the pipeline now answers in the language it was
asked in, and `language_match` reports the share for nothing, with no model call and no card time.

## Files

`datasets/measurements/language_directive_arc4_20260908.json`,
`datasets/measurements/language_costs_on_our_axes_arc4_20260908.json`,
`datasets/measurements/language_match_before_arc4_20260908.json`,
`datasets/measurements/language_cost_arc4_lang_after_20260908.json` (both cuts, and the code that
produces them: `app/evals/language_cost.py`, which reproduces the hand computation above)
