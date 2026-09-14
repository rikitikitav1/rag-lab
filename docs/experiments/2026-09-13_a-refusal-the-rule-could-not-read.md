# 2026-09-13 - A refusal the rule could not read: "the context does not contain"

Every answer gets an outcome, and the outcome is read from the answer's text on every read. The
refusal markers held "не содержится" but not "не содержит", and no "does not contain" in English, so
short answers like "Контекст не содержит информации об ионной связи" were counted as answers. This
entry records what fixing that moved, backwards through every recorded run, since nothing is stored
that would pin the old reading.

## Setup

**Set** the grid's four rejudged copies (`arc5_rejudge_grid_{a,b}_{llama,deepseek}_repeat=nows`),
300 rows each: 100 `paraphrased_ru` from the corpus, 100 `out_of_corpus` (87 English, 13 Russian),
100 `off_domain` (76 Russian, 24 English) · **corpus** `clean_1024`, `single_shot` · **generators**
`llama3.1:8b` on ollama and `deepseek-ai/DeepSeek-V4-Flash-0731` through gonka, two repeats each ·
**judge** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM, JSON without free whitespace

Only the reading rule varies, and no model was called. Rule 2 adds one pattern (`SOURCES_LACK` in
`app/outcomes.py`): "не содержит" or "does not contain" counts as a refusal only when its subject is
the source (context, sources, materials, documents, fragments, passages). The ceiling of 400
characters under which a refusal marker counts is unchanged. Rule 1's numbers were reproduced by
substituting an empty pattern.

The rule was written after the answers had been read: the hole was found while reading the refusals
of another comparison, so this is a correction of the instrument, not an A/B with a rule set in
advance.

## Result

Counts per 100 rows of each pool, rule 1 → rule 2.

| copy | off domain, refused | out of corpus, refused correctly | in corpus, false refusal | judge's means |
|---|---|---|---|---|
| llama, a | 56 → 57 | 7 → 12 | 0 → 0 | unchanged |
| DeepSeek, a | 71 → 93 | 24 → 59 | 0 → 2 | unchanged |
| llama, b | 54 → 58 | 8 → 16 | 0 → 0 | unchanged |
| DeepSeek, b | 72 → 86 | 23 → 59 | 1 → 2 | unchanged |

On the engine pair (the same generator on vLLM and on ollama) false refusals moved 4 → 5 and 0 → 1,
means again unchanged. The counts are exact readings of fixed text under two rules. The gap between
the two generators is a point estimate on 100 questions per pool, with the spread between repeats
(the a and b rows) as its only measure of noise.

## What moved and what did not

- Off domain, DeepSeek refuses 28 to 36 points more often than llama, not 15 to 18 as read under
  rule 1. Out of corpus the gap is larger still.
- The grid's reading on corpus questions stands. DeepSeek's lower relevance there comes from long
  hedged answers that open with "the context does not contain" and then answer anyway; they run past
  400 characters and are read as answers under both rules.
- The judge's means do not move, because the judge abstains on the refusal flag the answering path
  wrote at answer time, not on the outcome read now.

## Decision

- Rule 2 is the reading rule. Every report now names the rule it read its outcomes with
  (`outcome_rule`, currently 2), in the reader's reports rather than in the run's snapshot, because
  the outcome is computed at read time. Commit "Read a short answer whose sources do not
  contain the answer as a refusal, and name the refusal rule every report read its outcomes with".
- Misses of rule 2 are collected and change the pattern at once, as rule 3, rather than one by one.
- Not taken: a third outcome for "a hedge, then an answer". The judge's relevance already charges
  such rows (1.8 to 2.0 on the grid); a separate outcome needs a classifier and a reason to want one.

## Caveats

- A refusal share quoted before 13.09 was read with rule 1. A report without `outcome_rule` is
  rule 1.
- Known and left as is: the refusal flag on the row is written at answer time under the rule of that
  moment, while the outcome is read under the current one. Which rows the judge owes follows the
  former, so a few rows now read as refusals were still scored. Means do not move on the grid.
- The pattern is a few subject nouns in two languages. A refusal phrased another way ("I found
  nothing about...") stays unread until rule 3.
- A refusal longer than 400 characters is read as an answer under both rules.
