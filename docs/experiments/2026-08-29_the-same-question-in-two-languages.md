# 2026-08-29 - The same question in two languages

The generator grid ran the same interview questions twice: once paraphrased into Russian, once
paraphrased in English. Both sets descend from the same parents in `interview`, so the two halves
can be paired question by question and asked what a language costs. This reading was not
pre-registered; it was taken after the numbers existed, and it is written down as post-hoc.

## Setup

**Sets** `paraphrased_v2_ru` (n=823) and `paraphrased_v2` (n=820), paired through
`questions.source_question_id`, 820 pairs · **corpus** `clean_1024`, English throughout ·
**pipeline** `single_shot` · **judge** `qwen2.5:7b`, prompts v2

The arms are the grid's own, no new runs were made. Code: `temp_files/model_grid_cross.py`,
same pairing and the same eight-seed intervals as `retrieval_compare.bootstrap_ci`.

The corpus is English and the question is Russian, so a Russian answer is also a translation. One
judge scores both halves.

## Result

Russian minus English, paired on the shared parent, n=820. Positive means the Russian side scored
higher.

| arm | faithfulness | relevance | completeness |
|---|---|---|---|
| gemma plain | -0.7720 **direction only** | +0.0683 [-0.1610, +0.3061] | -0.0695 [-0.2329, +0.1024] |
| gemma rerank | -0.4415 **direction only** | +0.2902 [+0.0780, +0.5207] | +0.1305 [-0.0195, +0.2890] |
| llama plain | -0.9390 **direction only** | -0.4049 [-0.5878, -0.2280] | -0.1841 [-0.3427, -0.0195] |
| llama rerank | -0.8024 **direction only** | -0.2268 [-0.3866, -0.0549] | 0.0000 [-0.1476, +0.1366] |

`direction only` because the control arm of the grid measured a drift of +0.2442 on faithfulness
between two identical runs; see
[the generator grid](2026-08-29_generator-grid-4b-against-8b.md). The interval printed here is the
bootstrap over the pairing and does not include that drift.

Asking in Russian costs about a point of faithfulness for both models, while completeness stays
roughly level and relevance splits: gemma holds it, llama loses it.

## Two candidates, and this grid separates neither

**Translation drift.** The context is English, the answer is Russian, and the translation is where
a model comes loose from its source. That is exactly the defect faithfulness is built to catch.

**A blind spot in the judge.** `judge.completeness.v2` carries the clause "judge by meaning, not
wording, and across languages". `judge.faithfulness.v2` and `judge.relevance.v2` do not. The order
of the damage matches the order of the clause: completeness, which has it, barely moves; relevance
and faithfulness, which do not, move most. That is a coincidence across three prompts, which is a
reason to run a test, not a reason to draw a conclusion.

The discriminator is one run and it is planned: judge the same Russian answers with the same judge
and a faithfulness prompt that is v2 plus that clause, paired. If the gap narrows by more than the
judge's own noise, the blind spot is real; if it stays, the translation is. The judge's own noise
comes from the first arm of the re-judge stand, the same run judged twice.

## The gap narrowed for llama, and the reason is not what it looks like

Turning reranking on narrows llama's Russian faithfulness gap from -0.939 to -0.802. That reads as
"reranking helps the Russian side". The two sides say otherwise:

```
llama faithfulness, arm means
  ru  7.39 plain -> 7.35 rerank   (-0.04)
  en  8.33 plain -> 8.16 rerank   (-0.17)
```

The gap closed by 0.137 because the English side fell by 0.17, not because the Russian side rose.
Both moves are inside the drift the control arm measured, so neither is a result on its own; what
survives is the warning that a difference of differences can move for the wrong reason.

## Decision

Nothing is decided from this entry. It supplies the target for a judge-prompt experiment on the
re-judge stand, and it is the reason the first arm of that stand measures the judge's own noise
before anything else is read.

## Caveats

- **Post-hoc.** No decision rule was written before this reading, and the split by language was not
  among the six pre-registered predictions.
- Faithfulness magnitudes are unreadable at this stand's noise level. Only the direction is used.
- The pairing loses three of the 823 Russian questions, whose English siblings do not exist.
- "Russian costs a point" mixes the model and the judge by construction. It cannot be reported as a
  property of the generator.
- The corpus is English in both halves. Nothing here says what would happen with a Russian corpus.
