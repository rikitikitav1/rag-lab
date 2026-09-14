# 2026-09-14 - A cloud judge on the same answers, two and a half points kinder on grounding

Our judge is a 7B Qwen, and every comparison on this stand is read through it. A much larger cloud
model judging the very same answers says how much of what we read is the judge: where the two agree,
where one is kinder, and whether the cloud model is kinder to answers written by itself. The answers
are fixed, so only the judge moves.

## Setup

**Answers** the grid's repeat A, rejudged copies of `arc3_agent_baseline`'s 300 questions (100
`paraphrased_ru` from the corpus, 100 `out_of_corpus`, 100 `off_domain`), corpus `clean_1024`,
`single_shot`: answers by `llama3.1:8b` (copy `arc5_rejudge_grid_a_llama_repeat=nows`) and by
`deepseek-ai/DeepSeek-V4-Flash-0731` (copy `arc5_rejudge_grid_a_deepseek_repeat=nows`) · **our
judge** `Qwen/Qwen2.5-7B-Instruct-AWQ` on vLLM, JSON without free whitespace · **cloud judge**
`deepseek-ai/DeepSeek-V4-Flash-0731` through the gonka broker, the same prompts
(`judge_faithfulness` 2, `judge_relevance` 2, `judge_completeness` 2), `temperature` 0, `seed` 0,
`max_tokens` 8192, no JSON grammar on the broker's side · experiments 61 (llama's answers), 62
(DeepSeek's answers) and 63 (llama's answers again, for the cloud judge's own floor), kind
`rejudge`, equal `answers_digest` on every pair, Holm over the three axes

No numeric prediction was written before these runs. The owner's plan named two questions: how the
cloud judge compares with ours, and whether it is kinder to its own answers.

## Result

Cloud judge minus ours, paired over the same answers.

| answers | axis | n | Δ [95% CI] | p | Holm |
|---|---|---|---|---|---|
| llama (61) | faithfulness | 236 | +2.50 [+1.90, +3.11] | 5e-14 | passes |
| llama (61) | relevance | 237 | -0.97 [-1.25, -0.69] | 6e-11 | passes |
| llama (61) | completeness | 100 | -0.62 [-1.06, -0.20] | 0.005 | passes |
| llama again (63) | faithfulness | 237 | +2.54 [+1.95, +3.13] | 1e-14 | passes |
| llama again (63) | relevance | 237 | -0.89 [-1.18, -0.62] | 2e-9 | passes |
| llama again (63) | completeness | 100 | -0.44 [-0.88, -0.02] | 0.037 | passes |
| DeepSeek (62) | faithfulness | 205 | +2.60 [+2.13, +3.12] | 2e-20 | passes |
| DeepSeek (62) | relevance | 205 | -0.28 [-0.56, -0.03] | 0.059 | does not |
| DeepSeek (62) | completeness | 100 | +0.11 [-0.44, +0.62] | 0.42 | does not |

On DeepSeek's answers the relevance interval just clears zero while the paired test does not pass;
it is read as not shown.

## The cloud judge's own floor

The same llama answers judged twice by the cloud model (61 against 63): 154 of 573 verdicts
differ, 26.9% (faithfulness 54 of 236, relevance 57 of 237, completeness 43 of 100). The means stay
within 0.18 of each other; on the corpus pool completeness moved +0.18 [+0.02, +0.35], faithfulness
and relevance crossed zero. Our judge on vLLM moved none of its verdicts in its own repeats: 0 of 388
across a reload ([the batch invariance entry](2026-09-09_what-batch-invariance-costs-on-an-awq-judge.md))
and 0 of 275 within one residency under today's JSON rule
([the grammar entry](2026-09-13_the-judge-that-looped-on-whitespace.md)).

This floor was taken between two windows, not within one: 61 was judged in pieces, before 17:33 on
13.09 until a restart of the worker stopped it, a little after 23:22 until the broker's daily limit
stopped it, and the rest after 00:01 on 14.09; 63 ran between 17:35 and 19:13 and around 22:00. A broker keeps its answers repeatable within a window and
not across windows, so 26.9% is an upper bound of the floor within one window, which was not
measured.

## Reading

- **Kinder on grounding, by the same amount on every answer.** +2.50, +2.54 and +2.60: the same
  shift on llama's answers, on their repeat and on DeepSeek's own. It is about fourteen times the
  largest shift of the cloud judge's own means between its two passes (0.18), and it does not
  depend on whose answer it reads. The two judges hold different rulers for faithfulness; a number on one does not translate
  to the other.
- **Stricter on relevance and completeness to llama, not to itself.** -0.97 and -0.62 on llama's
  answers against -0.28 and +0.11 (neither shown) on its own. That looks like a judge that spares
  its own answers, but the two sets of answers differ: DeepSeek's are shorter and several of them
  decline inside the answer ([the grid entry](2026-09-13_a-bigger-model-at-the-same-retrieval.md)).
  So "kinder to itself" is a hint on this material, not a finding.
- **As an instrument for reading rows, the cloud judge is worse than ours.** A quarter of its
  verdicts move between two passes; ours move none. For means over 100 rows it is usable.

## Decision

- Our judge stays the default. The cloud judge is a second opinion on means, not a ruler for rows.
- A difference read with one judge is not carried to the other: faithfulness in particular sits 2.5
  points apart.
- Whether the cloud judge spares its own answers needs two answer sets that differ only by author,
  which this grid does not give.

## Caveats

- One judge prompt version on both sides, written for and tuned against the 7B judge; the cloud
  model reads it as it is.
- The cloud judge's floor is an upper bound across windows, as above.
- 100 corpus questions per answer set, Russian paraphrases only; faithfulness and relevance also
  cover the scorable rows outside the corpus.
- The broker decides the cloud model's weights, precision and node, and nothing the stand reads can
  tell whether they changed between the passes.
- Answers where the stand refused before any generator call are stamped with the default generator
  on DeepSeek's copy (38 off-domain rows). They are refusals and carry no judged axis, so the table
  above is not touched by it.
