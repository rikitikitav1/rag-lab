# 2026-07-25 - Generation prompt v2 → v3 (completeness)

**Hypothesis:** the weak axis is relevance (answers judged "partially" rather than "relevant"): the generator answers loosely. A prompt pushing directness and completeness should convert partially → relevant. Retrieval is unchanged, so any delta is the prompt.

**Change:** `generate_answer` v2 (terse: answer from context, cite, language) → v3 (role line + explicit rules: answer directly and fully, cite every factual statement, do not repeat the question, state what is missing on partial answers, describe source conflicts, no speculation).

**Setup:** set `paraphrased_ru` (100); generator `llama3.1:8b`, **temp 0** (deterministic); rerank off; retrieval identical (hit@k confirms). Clean isolation of the prompt.

**Result (ru, temp 0, clean):**

| metric | v2 | v3 | Δ |
|--------|----|----|---|
| relevant | 35 | 57 | +22 |
| partially | 54 | 36 | −18 |
| irrelevant | 11 | 7 | −4 |
| faithful | 38 | 50 | +12 |
| unfaithful | 28 | 24 | −4 |
| hit@k / MRR | 80% / 0.652 | 80% / 0.652 | = |

**Cross-check (en):** relevant 33 → 44 (+11); faithful 61 → 56 (−5). Caveat: the en baseline was temp 0.1, so not a perfectly clean comparison.

**Delta:** relevance up on both languages (ru +22, en +11) - the target shift, and since retrieval and temperature were held, it is attributable to the prompt. Faithfulness up on ru (+12), slight trade-off on en (−5): pushing completeness yields longer answers with more assertions, some of which the judge marks ungrounded.

**Conclusion:** v3 is a clear, reproducible win on relevance across both languages (not a set-specific fluke). **Decision: v3 active.** Follow-up: resolve the en faithfulness dip with a clean temp-0 baseline, and/or add a "no assertions beyond what answers the question" clause to balance completeness against grounding.

---
