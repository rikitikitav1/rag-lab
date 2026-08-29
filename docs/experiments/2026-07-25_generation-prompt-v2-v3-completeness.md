# 2026-07-25 - Generation prompt v2 → v3 (completeness)

The weak axis is relevance: answers land in "partially" rather than "relevant", which reads as a
generator answering loosely. A prompt that pushes directness and completeness should convert
partially into relevant. Retrieval does not change, so any delta belongs to the prompt.

## Setup

**Set** `paraphrased_ru` (n=100), cross-checked on `paraphrased` (n=100) · **corpus** pre-variant era, `developer-roadmap` still in · **judge** `qwen2.5:7b`, categorical

`generate_answer` v2 (terse: answer from context, cite, language) against v3 (role line plus
explicit rules: answer directly and fully, cite every factual statement, do not repeat the
question, state what is missing on partial answers, describe source conflicts, no speculation).

Generator `llama3.1:8b` at temp 0, rerank off. Retrieval is identical in both arms, confirmed by
hit@k.

## Result

| metric | v2 | v3 | Δ |
|--------|----|----|---|
| relevant | 35 | 57 | +22 |
| partially | 54 | 36 | −18 |
| irrelevant | 11 | 7 | −4 |
| faithful | 38 | 50 | +12 |
| unfaithful | 28 | 24 | −4 |
| hit@k / MRR | 80% / 0.652 | 80% / 0.652 | = |

Cross-check on the English set: relevant 33 → 44, faithful 61 → 56.

Relevance rises on both languages, which is the targeted shift, and retrieval and temperature were
held, so it is attributable to the prompt. Faithfulness rises on ru and dips on en: pushing
completeness yields longer answers with more assertions, some of which the judge marks ungrounded.

## Decision

v3 active. Follow-up: a clean temp-0 en baseline for the faithfulness dip, and possibly a "no
assertions beyond what answers the question" clause to balance completeness against grounding.

## Caveats

- the en baseline ran at temp 0.1, so the en column is not a clean comparison
- counts, not scores: this run predates the numeric judge, and the categorical counter was later
  shown to invert conclusions ([the k-sweep re-judge](2026-07-28_k-sweep-re-judged-with-a.md))
- single run per arm, no interval
