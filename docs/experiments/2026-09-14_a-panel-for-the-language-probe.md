# 2026-09-14 - A panel for the language probe, and a regime read against a reference

The language probe measures what our judge charges for Russian: it restates a row's answer in English
and in Russian and judges both. Its control said whether the judge was "in regime", and it read that
from the run's own rows at a fixed share of 0.90. A Russian run pulled that share down through the
judge's weak Russian, so the probe was calibrated by the very thing it measures. This entry records
the fix: a fixed panel, and a regime read against a reference reading of it.

## Setup

**Panel** `datasets/measurements/judge_language_panel_ids.txt`: first 30, then all 60 English
corpus rows of `paraphrased_single_shot_1788245198_variant_clean_1024` (set `paraphrased`, corpus
`clean_1024`) answered with sources, chosen by no score · **judge** `Qwen/Qwen2.5-7B-Instruct-AWQ` on
vLLM, JSON without free whitespace · **runs** `judge_language` on 5 rows of `arc3_agent_baseline`,
jobs 2842 (30-row panel, schema 6) and 2844 (60-row panel, schema 7), 481 s and 918 s

Each run judges every panel row twice: its own answer, and that answer restated in English. The
first gives the judge's regime, the second the restating path's.

## Result

| run | panel | own answers at least 7 | restated at least 7 | against 0.90 |
|---|---|---|---|---|
| 2842 | 30 rows | 26 of 30, 0.867 | 27 of 30, 0.90 | below on own answers |
| 2844 | 60 rows | 55 of 60, 0.917 | 57 of 60, 0.95 | above on both |

At 30 rows one row flipped the regime. Two of the panel's answers are weak under any judge: the old
judge (`qwen2.5:7b` on ollama) scored rows 33263 and 33270 at 0 and 2 when their run was recorded, and
the new judge scored them low in both runs. The ceiling of a 30-row panel is therefore about 0.93,
one row above the threshold. Choosing the panel by score would have removed them, and would have had
the instrument pick its own test, so the panel grew instead.

## Decision

- The panel is all 60 rows, and every panel verdict is written into the probe's report
  (`panel_rows`: row, part, score, reason).
- 0.90 stays as a floor that catches only the gross (`above_floor`). The regime is read against a
  reference reading instead: `moved_vs_reference` counts rows that crossed 7 in either direction
  against the reference, a lost verdict included, and `in_regime` holds while at most 6 of 60 moved
  on each part. Without a reference the report says `in_regime: null` and why.
- The reading of 2844 became the reference on the owner's word
  (`datasets/measurements/judge_language_panel_reference.json`, taken 14.09, with the instrument's
  stamp). Checked against itself: 0 rows moved, `in_regime: true`. It is rewritten only when the ruler
  changes: the grammar, the penalty, the server's version.
- The run's own rows keep their share as `run_answers`, a property of the run, without the word
  regime. Commit "Read the language probe's regime on a fixed panel of sixty English rows against a
  reference reading, and keep its readings".

## Caveats

- 60 rows give a step of one sixtieth; six rows is the allowance, and a panel that hovers near it
  should grow again rather than loosen it.
- The panel is English only. It says whether the judge holds its ruler, not what the ruler charges
  for Russian, which is still the probe's own measurement.
- The reference is one reading. Its own spread against a second reading in another load was not
  measured, so the allowance of six is a rule, not a floor.
