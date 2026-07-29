# 2026-07-27 - Generation prompt v3 → v4 (drop inline citations)

**Hypothesis:** the three citation rules in v3 buy nothing that the structured `sources` field does not already carry, and they cost answer cleanliness. Inline `[source]` markers appeared unevenly across runs (present on some single-shot answers, absent on others), so as a consumer-facing contract they were unreliable anyway. Dropping them should leave grounding untouched, since the load-bearing rules are "use only information explicitly supported by the context" and the partial/absent/conflict clauses, not the citation lines.

**Change:** `generate_answer` v3 → v4: removed exactly three lines ("cite every factual statement in the format [source]", "do not cite a source unless it directly supports the statement", "cite all relevant sources when several support a statement"). Every other rule is byte-identical to v3.

**Scope:** `generate_answer` is read only by `chat.answer` (`app/use_cases/chat.py:180`), so this affects **single_shot only**. The agent synthesizes through `agent_system` in its ReAct loop and is untouched; `prompts/agent_system.v2.txt:11` still instructs the agent to cite inline. That asymmetry is deliberate for now: changing the agent prompt would need its own agent eval, and folding it into this run would confound the two.

**Judge impact:** none by construction. `judge_faithfulness.v1`, `judge_relevance.v1` and `judge_completeness.v1` contain no reference to citations or `[source]` markers, so the rubric does not move. v3 and v4 numbers stay comparable on all three axes.

**Setup:** set `paraphrased_ru` (100); generator `llama3.1:8b`, temp 0; rerank off. Measured on the current (roadmap-removed) corpus, so a v3 control (`v3_current`) was run alongside v4 (`v4_ru`) on the same corpus. Retrieval is prompt-independent and came out identical for both (91% / MRR 0.837), confirming a clean single-variable A/B. That retrieval is higher than the pre-roadmap-drop `v3_ru` baseline above (80% / 0.652), which is the corpus change, not the prompt.

**Result (v3 with citations vs v4 without, same corpus):**

| axis | v3 (inline citations) | v4 (no citations) |
|------|-----------------------|-------------------|
| faithful / partially / unfaithful | **56 / 29 / 15** | 49 / 30 / 21 |
| relevant / partially / irrelevant | **68 / 28 / 4** | 56 / 35 / 9 |
| complete / partially / incomplete | 2 / 87 / **11** | 1 / 82 / 17 |

**v4 is worse on all three axes**, most on relevance (68 → 56, irrelevant 4 → 9) and faithfulness (unfaithful 15 → 21).

**Conclusion: the hypothesis was wrong; the citation rules were load-bearing, not cosmetic.** Forcing the 8B to attribute every statement to a source is a grounding discipline that keeps the answer tied to the retrieved context and on topic. Removing it let the model drift into less relevant, less grounded prose. The uneven inline markers were a symptom of a weak model following the instruction imperfectly, not a reason to drop it. **Decision: revert. v4 deleted (file and DB row), v3 stays active.** The consumer-facing sources contract is handled honestly at the API/doc layer instead: the structured `sources` list is the authoritative channel, and the answer prose may still carry inline `[source]` markers unevenly. Follow-up: consistent inline citation is a model-capability lever (a stronger generator), not a prompt-removal one. Measuring reverted a plausible cosmetic change before it shipped a quality regression.

---
