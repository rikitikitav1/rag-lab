# 2026-07-26 - ReAct agent vs single-shot (a measured loss)

**Hypothesis:** a ReAct agent that decides its own retrieval (search when needed, refine, multi-hop) should answer at least as well as single-shot RAG, and better on questions needing decomposition. New judge axis **completeness** (answer vs the question's reference answer) added specifically because faithfulness/relevance cannot see the agent's expected advantage.

**Change:** agentic pipeline (`agent.run`, ReAct loop over a `search_corpus` tool, `agent_system` prompt) vs single-shot (`chat.answer`, `generate_answer` prompt). Same `paraphrased_ru` set (100), same gold labels, same judge (qwen2.5:7b).

**Result:**

| axis | base_ru_t0 | v3_ru (single-shot) | agent_ru |
|------|-----------|---------------------|----------|
| retrieval hit@k | 80% | 80% | **31%** |
| retrieval MRR | 0.652 | 0.652 | **0.277** |
| faithful | 38 | 50 | **23** |
| unfaithful | 28 | 24 | **55** |
| relevant | 35 | 57 | 47 |
| complete / partially / incomplete | 0/73/27 | 1/85/14 | 1/72/**27** |
| avg hops | - | - | 2.03 (max 3) |

**Diagnosis (the point of the experiment):** the agent **lost on every axis**. Root cause is retrieval, not generation: **54/100 agent answers had zero retrieved sources**. Every question triggered a search (no run stopped at hop 1), so the agent did not skip retrieval - its self-formulated queries **returned empty ~54% of the time**, while single-shot embeds the full question and retrieves at 80%. Handing query formulation to the 8B model collapsed the cross-lingual retrieval that direct embedding gets for free (reformulated queries drift past the distance threshold). Empty context then propagates downstream: unfaithful jumps to 55 (the model answers ungrounded when the search comes back empty). Only 3/100 runs did 3 hops, so there is no decomposition benefit to offset the retrieval loss.

**Note on metrics:** retrieval hit@k/MRR here are computed the same way for both pipelines; for the agent this is really recall-across-hops, not precision@k, so it is not a clean head-to-head (see PROGRESS backlog). In this run the caveat did not matter - the agent barely multi-hops and its searches often return empty, so the number is deflated, not inflated. (An earlier worry that the agent would *inflate* hit@k via recall was wrong; measuring corrected it.)

**Conclusion:** on this workload (cross-lingual, mostly single-fact questions, 8B model, single retrieval tool) the naive ReAct agent is **strictly worse** than single-shot v3. Agentic RAG is not free: giving a weak model control over query formulation destroyed retrieval. **Decision: single-shot v3 stays the default.** Follow-ups (backlog): have the agent search with the original question (or add it alongside its reformulation), and/or relax the distance threshold for tool searches; the agent's value likely needs genuinely multi-part questions and a stronger model, not this set.

**Follow-up (same day) - the loss was a bug, not the agent (the diagnosis above was wrong).** Inspecting the actual tool-call arguments disproved the "query drift past the distance threshold" theory. The empties came from a different place: the 8B model **always filled the optional `category` parameter with a hallucinated ltree path** (`numpy.array_operations`, `databases.amazon`, `angularjs`), and `category ~ lquery` returns zero for paths that do not exist in the taxonomy. The query *text* the model produced was fine (full, on-topic). Even the rare non-empty case was wrong-domain (a `databases.redis` guess returned redis chunks for a reinforcement-learning question). **Fix: remove `category` from the tool schema offered to the model** (the `_search_corpus` function still supports it; it is just not exposed). General tool-design lesson: never offer a free-form parameter the model cannot ground - use an enum (small fixed set) or a discovery tool; here the taxonomy is too large for an enum, so drop it.

**Re-run (`agent_nocat_ru`, category removed):**

| axis | v3_ru (single-shot) | agent_ru (category trap) | agent_nocat_ru (fixed) |
|------|---------------------|--------------------------|------------------------|
| retrieval hit@k | 80% | 31% | **84%** |
| retrieval MRR | 0.652 | 0.277 | **0.773** |
| faithful | 50 | 23 | 37 |
| unfaithful | 24 | 55 | 32 |
| relevant | 57 | 47 | **70** |
| complete / partially / incomplete | 1/85/14 | 1/72/27 | **2/89/9** |
| empty-source answers | - | 54 | **0** |

**Revised conclusion:** "agent is strictly worse" was an artifact of the category bug, not a property of agentic RAG. With the trap removed the agent **beats single-shot on retrieval (84 vs 80, MRR 0.773 vs 0.652), relevance (70 vs 57), and completeness (incomplete 9 vs 14)** - its query reformulation (often ru→en) retrieves better cross-lingually than embedding the raw Russian question. The one axis it loses is **faithfulness (37 vs 50 faithful, 32 vs 24 unfaithful)**: answering more fully from a multi-hop context union yields more assertions the judge marks ungrounded (the completeness-vs-grounding tradeoff, plus a different context denominator than single-shot's top-3). **Takeaway: measuring caught a bug the naive read would have shipped as a conclusion.** Open follow-up: close the faithfulness gap (tighter grounding instruction; or judge the agent against a deduped/narrowed context).

**Post-remediation re-run (`agent_v2_ru`).** After a second review round the agent/judge path was hardened: `answered` now follows retrieved evidence (honest refusal on empty), the judged context excludes the "No relevant documents found." / tool-error sentinels, judging is per-axis, and `dispatch` drops any hallucinated `category`. Re-running the same set validated the fixes and moved the faithfulness gap the predicted follow-up direction:

| axis | v3_ru (single-shot) | agent_nocat_ru | agent_v2_ru (hardened) |
|------|---------------------|----------------|------------------------|
| retrieval hit@k | 80% | 84% | **87%** |
| retrieval MRR | 0.652 | 0.773 | **0.779** |
| faithful | 50 | 37 | 39 |
| unfaithful | 24 | 32 | **29** |
| relevant | 57 | 70 | 68 |
| complete / partially / incomplete | 1/85/14 | 2/89/9 | **5/90/5** |
| empty-source answers | - | 0 | 0 |

Filtering non-evidence out of the faithfulness context (the round-2 fix) is what nudged unfaithful 32 → 29 and faithful 37 → 39: the judge was previously scoring some answers against a context that literally said "No relevant documents found." The overall conclusion stands - the agent beats single-shot on retrieval, relevance, and completeness, and trails on faithfulness, with the gap now narrower. The fix improved measurement validity, not the model.

**Metric caveats (post round-3 hardening).** Three numbers here shifted meaning and are not strictly comparable across rounds: (1) agent MRR reflects cross-hop source dedup added in `24fae88`, so pre-dedup agent MRR (`agent_nocat_ru` 0.773) is not directly comparable to `agent_v2_ru` 0.779 (hit@k is unaffected - membership is unchanged). (2) The agent's `hops` now includes the forced synthesis turn, so a run capped at N can report N+1. (3) faithfulness and relevance are counted over in-corpus logs; completeness is counted over all logs (in-corpus plus the out-of-corpus refusal probes), so the three axes do not share a denominator.

---
