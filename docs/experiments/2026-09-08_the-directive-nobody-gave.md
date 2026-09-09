# 2026-09-08 - The directive nobody gave, and what our judge charges for Russian

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

Splitting by whether the row actually changed language is where it becomes a number:

```
changed language, 35 rows   faithfulness  -0.971 [-1.600, -0.371]   zero excluded
                            completeness  -0.714 [-1.429,  0.029]
                            relevance     -0.086 [-0.857,  0.743]

kept its language, 64 rows  faithfulness  +0.141 [-0.438,  0.719]
                            relevance     -0.312 [-0.812,  0.141]
                            completeness  -0.109 [-0.562,  0.312]
```

The whole of the faithfulness drop sits on the rows that switched into Russian. Where the language
did not move, the axis did not move.

## Minus 0.971 against 0.964

The day before, a probe measured the same thing by a different construction: it took one answer and
asked the model to restate it in two languages without changing what it said, then judged both. That
gave **0.964 [0.645, 1.282]**.

This run did not restate anything. These are different answers from different runs, grouped after
the fact by whether their language changed. Two constructions, one number: about a point out of ten,
charged by our faithfulness prompt for an answer arriving in the language it was asked in.

That is a replication rather than a second look at one measurement, with one caveat named here
because it is not small: the group of 35 was selected by the outcome of the change, not declared in
advance. The prediction was pre-registered; this split was not.

## What it means for everything measured before

The defect is in the judge, not in the pipeline, and it sits on the axis this stand uses to mean
"grounded". Every number taken here on a population mixed by language carries it in an unknown
proportion, including the agent against single-shot comparison, because neither arm was instructed
and both drifted at rates we had not measured.

Fixing the judge's prompt would end the comparability of everything judged so far, which is why it is
a decision and not a task. The cheap half is done: the pipeline now answers in the language it was
asked in, and `language_match` reports the share for nothing, with no model call and no card time.

## Files

`datasets/measurements/language_directive_arc4_20260908.json`,
`datasets/measurements/language_costs_on_our_axes_arc4_20260908.json`,
`datasets/measurements/language_match_before_arc4_20260908.json`
