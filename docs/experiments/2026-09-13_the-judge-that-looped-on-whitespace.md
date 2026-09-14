# 2026-09-13 - The judge that looped on whitespace, and a grammar rule that moved 17% of its verdicts

One row out of 2419 judge calls never got a faithfulness score: the judge wrote a short reason and
then whitespace until its token ceiling. The fix is a server rule that forbids free whitespace in the
judge's JSON. A rule on how the judge decodes is a change of instrument, so before any comparison
this entry asks how much it moves the verdicts on answers that did not change.

## Setup

**Set** source run `arc5_grid_a_model_llama3.1_8b`, 150 of its 300 rows drawn with `sample_seed` 0
(113 scorable verdicts on faithfulness and relevance, 49 on completeness) · **corpus** `clean_1024`,
`single_shot`, answers by `llama3.1:8b` · **judge** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM 0.29.1,
prompts `judge_faithfulness` 2, `judge_relevance` 2, `judge_completeness` 2, `temperature` 0,
`seed` 0 · experiment 54, `rejudge`, passes `repeat=a` and `repeat=b` on identical answers (equal
`answers_digest`)

What varies: the source run's own verdicts were decoded with free whitespace between JSON elements;
both new passes were decoded under the server flag
`--structured-outputs-config {"backend": "xgrammar", "disable_any_whitespace": true}`. The reading
rule was written before the run, in the arc's log: the old against the new counts as a change of
instrument if it moves more verdicts than the new judge moves against itself (a against b).

## The loop

Row 42359 (grid, DeepSeek, `paraphrased_ru`, a question about types in Julia, an answer of 1568
characters) failed faithfulness three attempts in a row with "no JSON object". Reproduced outside the
queue with the same prompt and sampler: a normal short `reason`, then instead of `,"score"` a
repeating tail of tabs and newlines up to the ceiling. `finish_reason` was `length` at 1024 tokens and
at 4096 alike, on an input of 1937 tokens. The grammar allows whitespace between elements, and at
temperature 0 the same choice repeats, so a higher ceiling does not help. Per-request
`structured_outputs` with `disable_any_whitespace` were accepted by the server and changed nothing;
only the server flag did.

Frequency: one loop in 2419 judge calls over the grid and its floors.

## Result

| comparison | verdicts moved | faithfulness | relevance | completeness |
|---|---|---|---|---|
| old rule against new (a) | 47 of 275, 17.1% | 28 of 113 | 16 of 113 | 3 of 49 |
| new against new (a against b), one residency | 0 of 275 | 0 of 113 | 0 of 113 | 0 of 49 |

The means do not move. Paired over the same rows, old against a: faithfulness +0.009
[-0.381, 0.389], relevance -0.018 [-0.115, 0.089], completeness +0.061 [-0.122, 0.245], none
significant under Holm over nine tests. Rows moved both ways almost equally (faithfulness 15 better,
13 worse; relevance 7 and 9; completeness 2 and 1).

## Why this is the grammar and not the reload

Pass a differs from the old verdicts in two things at once: a reload of the engine and the rule. The
same judge on vLLM moved 0 of 388 verdicts across a reload
([the batch invariance entry](2026-09-09_what-batch-invariance-costs-on-an-awq-judge.md)), so the 17%
belongs to the rule. `compare_pools` says the same on its own: the pair carries
`one_judge_grammar: false` and "this contrast measures the grammar".

## Means stand, rows scatter

A symmetric scatter leaves means on 113 rows in place and breaks every per-row pair. On the grid's
four arms the rule also interacts with answer style: rejudged under it, DeepSeek's faithfulness moved
-0.31 in both repeats (raw p 0.038 in one) and its completeness +0.20 in one (passes Holm), while
llama moved nothing significant. So a comparison of two arms judged under different rules is a
comparison of rules, whatever the means say.

## Decision

- The flag is the judge's default on the `vllm` and `vllm-cpu` services (`docker-compose.yml`).
- The grid's four arms were rejudged by the new judge as copies (experiments 55-58). The grid's
  original verdicts stay as the record of the old ruler; every comparison of the grid reads the
  copies. The row that looped scored faithfulness 10 on its copy.
- A reply cut by the output ceiling before its score is recorded as such (`JudgeCut`,
  `judge_cut_by_length` on the axis), where `compare_pools` used to show zero cuts. The rule a verdict
  was decoded by is stamped on the row (`json_backend`, `json_disable_any_whitespace`), and a contrast
  across two rules is flagged. Commit "Stamp the judge's instrument whole: where a
  model sat after its call, the dtype vLLM loaded, JSON without free whitespace, a reply cut before
  its score, and a sweep's last attempt".
- Not taken: per-request `structured_outputs` (accepted, no effect) and forbidding whitespace only on
  a retry after `length`. The second would have been a per-request setting too, and per-request had no
  effect, so a retry-only rule had nothing to switch.

## Caveats

- 150 rows of one source run, one judge model, one engine. The 17% belongs to this pair and these
  prompts; another judge would move by another share.
- The loop was seen once in 2419 calls. This entry shows the row that looped now scores, not that the
  rule removes every loop.
- The new judge's floor here is within one residency. Across a reload it is carried from the entry
  linked above, which measured the old rule, and was not measured again under the new one.
- Runs judged under the old rule (the grid's originals, the cloud floor's two passes) stay comparable
  among themselves, not with anything judged after 13.09.
